/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 by Wolfgang Bangerth and SAATI Co.
 *
 * ---------------------------------------------------------------------
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

// The two most interesting header files will be these two:
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/meshworker/mesh_loop.h>
// The first of these is responsible for providing the class FEInterfaceValue
// that can be used to evaluate quantities such as the jump or average
// of shape functions (or their gradients) across interfaces between cells.
// This class will be quite useful in evaluating the penalty terms that appear
// in the C0IP formulation.


#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <complex>


namespace MembraneOscillation
{
  using namespace dealii;

  using ScalarType = std::complex<double>;


  // The following namespace defines material parameters. We use SI units.
  namespace MaterialParameters
  {
    double domain_extent;
    double density;
    double thickness;
    ScalarType tension;
    ScalarType stiffness_D;

    double min_omega = 100*2*numbers::PI;
    double max_omega = 10000*2*numbers::PI;
    unsigned int n_frequencies = 100;
  }


  void
  declare_parameters (ParameterHandler &prm)
  {
    prm.declare_entry ("Domain edge length", "0.015",
                       Patterns::Double(0),
                       "The edge length of the square domain. Units: [m].");
    prm.declare_entry ("Thickness", "0.0001",
                       Patterns::Double(0),
                       "Thickness of the membrane. Units: [m].");
    prm.declare_entry ("Density", "100",
                       Patterns::Double(0),
                       "Volumetric density of the membrane material. Units: [kg/m^3].");
    prm.declare_entry ("Loss angle", "2",
                       Patterns::Double(0,90),
                       "The angle used to make the Young's modulus complex-valued. "
                       "Units: [degrees].");
    prm.declare_entry ("Young's modulus", "200e6",
                       Patterns::Double(0),
                       "The magnitude of the Young's modulus. Units: [Pa].");
    prm.declare_entry ("Poisson's ratio", "0.3",
                       Patterns::Double(0,0.5),
                       "Poisson's ratio. Units: none.");
    prm.declare_entry ("Tension", "30",
                       Patterns::Double(0),
                       "The tension coefficient T that describes the membrane part of "
                       "the material behavior. Units: [N/m].");

    prm.declare_entry ("Minimal frequency", "100",
                       Patterns::Double(0),
                       "The minimal frequency to consider. Units: [Hz].");    
    prm.declare_entry ("Maximal frequency", "10000",
                       Patterns::Double(0),
                       "The maximal frequency to consider. Units: [Hz].");
    prm.declare_entry ("Number of frequency steps", "100",
                       Patterns::Integer(1,10000),
                       "The number of frequency steps into which the frequency "
                       "range is to be subdivided. Units: none.");
  }


  void
  read_parameters (ParameterHandler &prm)
  {
    // First read parameter values from the input file 'biharmonic.prm'
    prm.parse_input ("biharmonic.prm");
    

    
    using namespace MaterialParameters;
    
    // First get the independent parameters from the input file:
    double loss_angle, E, poissons_ratio;
    
    domain_extent  = prm.get_double ("Domain edge length");
    thickness      = prm.get_double ("Thickness");
    density        = prm.get_double ("Density");
    loss_angle     = prm.get_double ("Loss angle");
    E              = prm.get_double ("Young's modulus");
    poissons_ratio = prm.get_double ("Poisson's ratio");
    tension        = prm.get_double ("Tension");

    min_omega = prm.get_double ("Minimal frequency") * 2 * numbers::PI;
    max_omega = prm.get_double ("Maximal frequency") * 2 * numbers::PI;
    n_frequencies = prm.get_integer ("Number of frequency steps");
    

    // Then compute the dependent ones. Note that we interpret the angle in degrees.
    const ScalarType youngs_modulus
      = E * std::exp(std::complex<double>(0,2*numbers::PI*loss_angle/360));

    stiffness_D
      = (youngs_modulus *
         ScalarType(thickness * thickness * thickness
                    / 12 / (1 - poissons_ratio * poissons_ratio)));
  }
  

  

  
  // A data structure that is used to collect the results of the computations
  // for one frequency. The main class fills this for a given frequency
  // in various places of its member functions, and at the end puts it into
  // a global map.
  struct OutputData
  {
    ScalarType normalized_amplitude_integral;
    double normalized_maximum_amplitude;

    std::string visualization_file_name;
  };
  
    
  
  // A variable that will collect the data (value) for all frequencies
  // omega (key). Since we will access it from different threads, we also
  // need a mutex to guard access to it.
  std::map<double,OutputData> results;
  std::mutex results_mutex;


  // Return whether an external program has left a signal that
  // indicates that the current program run should terminate without
  // computing any further frequency responses. This is done by
  // placing the word "STOP" into the file "termination_signal" in the
  // current directory.
  bool check_for_termination_signal()
  {
    static bool termination_requested = false;

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock_guard (mutex);

    if (termination_requested == true)
      return true;
    
    // Try and see whether we can open the file at all. If we can't,
    // then no termination signal has been sent. If so, return 'true',
    // but before that set a flag that ensures we don't have to do the
    // expensive test with the file in any further calls
    std::ifstream in("termination_signal");
    if (!in)
      {
        termination_requested = false;
        return false;
      }

    // OK, the file exists, but does it contain the right content?
    std::string line;
    std::getline(in, line);
    if (line == "STOP")
      {
        termination_requested = true;
        return true;
      }

    // The file exists, but it has the wrong content (or no content so
    // far). This means no termination.
    return false;
  }

  

  template <int dim>
  class RightHandSide : public Function<dim>
  {
    public:
      static_assert(dim == 2, "Only dim==2 is implemented");

      RightHandSide (const double omega)
      : omega (omega)
      {}

      virtual double value(const Point<dim>  &/*p*/,
                           const unsigned int /*component*/ = 0) const override

      {
        return 1;
      }

    private:
      double omega;
  };



  // @sect3{The main class}
  //
  // The following is the principal class of this tutorial program. It has
  // the structure of many of the other tutorial programs and there should
  // really be nothing particularly surprising about its contents or
  // the constructor that follows it.
  template <int dim>
  class BiharmonicProblem
  {
  public:
    BiharmonicProblem(const unsigned int fe_degree,
                      const double omega);

    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void postprocess();
    void output_results();

    // The frequency that this instance of the class is supposed to solve for.
    const double                  omega;

    Triangulation<dim>            triangulation;

    MappingQ<dim>                 mapping;

    FE_Q<dim>                     fe;
    DoFHandler<dim>               dof_handler;
    AffineConstraints<ScalarType> constraints;

    SparsityPattern               sparsity_pattern;
    SparseMatrix<ScalarType>      system_matrix;

    Vector<ScalarType>            solution;
    Vector<ScalarType>            system_rhs;

    OutputData                    output_data;
  };



  template <int dim>
  BiharmonicProblem<dim>::BiharmonicProblem(const unsigned int fe_degree,
                                            const double omega)
    : omega (omega)
	, mapping(1)
    , fe(fe_degree)
    , dof_handler(triangulation)
  {}



  // Next up are the functions that create the initial mesh (a once refined
  // unit square) and set up the constraints, vectors, and matrices on
  // each mesh. Again, both of these are essentially unchanged from many
  // previous tutorial programs.
  template <int dim>
  void BiharmonicProblem<dim>::make_grid()
  {
    GridGenerator::hyper_cube(triangulation, 0, MaterialParameters::domain_extent);
    triangulation.refine_global(5);
  }



  template <int dim>
  void BiharmonicProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             ZeroFunction<dim,ScalarType>(),
                                             constraints);
    constraints.close();


    DynamicSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         c_sparsity,
                                         constraints,
                                         true);
    sparsity_pattern.copy_from(c_sparsity);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }



  // @sect{Assembling the linear system}
  //
  // The following pieces of code are more interesting. They all relate to the
  // assembly of the linear system. While assemling the cell-interior terms
  // is not of great difficulty -- that works in essence like the assembly
  // of the corresponding terms of the Laplace equation, and you have seen
  // how this works in step-4 or step-6, for example -- the difficulty
  // is with the penalty terms in the formulation. These require the evaluation
  // of gradients of shape functions at interfaces of cells. At the least,
  // one would therefore need to use two FEFaceValues objects, but if one of the
  // two sides is adaptively refined, then one actually needs an FEFaceValues
  // and one FESubfaceValues objects; one also needs to keep track which
  // shape functions live where, and finally we need to ensure that every
  // face is visited only once. All of this is a substantial overhead to the
  // logic we really want to implement (namely the penalty terms in the
  // bilinear form). As a consequence, we will make use of the
  // FEInterfaceValues class -- a helper class in deal.II that allows us
  // to abstract away the two FEFaceValues or FESubfaceValues objects and
  // directly access what we really care about: jumps, averages, etc.
  //
  // But this doesn't yet solve our problem of having to keep track of
  // which faces we have already visited when we loop over all cells and
  // all of their faces. To make this process simpler, we use the
  // MeshWorker::mesh_loop() function that provides a simple interface
  // for this task: Based on the ideas outlined in the WorkStream
  // namespace documentation, MeshWorker::mesh_loop() requires three
  // functions that do work on cells, interior faces, and boundary
  // faces; these functions work on scratch objects for intermediate
  // results, and then copy the result of their computations into
  // copy data objects from where a copier function copies them into
  // the global matrix and right hand side objects.
  //
  // The following structures then provide the scratch and copy objects
  // that are necessary for this approach. You may look up the WorkStream
  // namespace as well as the
  // @ref threads "Parallel computing with multiple processors"
  // module for more information on how they typically work.
  template <int dim>
  struct ScratchData
  {
    ScratchData(const Mapping<dim> &      mapping,
                const FiniteElement<dim> &fe,
                const unsigned int        quadrature_degree,
                const UpdateFlags         update_flags = update_values |
                                                 update_gradients |
                                                 update_quadrature_points |
                                                 update_JxW_values,
                const UpdateFlags interface_update_flags =
                  update_values | update_gradients | update_quadrature_points |
                  update_JxW_values | update_normal_vectors)
      : fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags)
      , fe_interface_values(mapping,
                            fe,
                            QGauss<dim - 1>(quadrature_degree),
                            interface_update_flags)
    {}


    ScratchData(const ScratchData<dim> &scratch_data)
      : fe_values(scratch_data.fe_values.get_mapping(),
                  scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags())
      , fe_interface_values(scratch_data.fe_values.get_mapping(),
                            scratch_data.fe_values.get_fe(),
                            scratch_data.fe_interface_values.get_quadrature(),
                            scratch_data.fe_interface_values.get_update_flags())
    {}

    FEValues<dim>          fe_values;
    FEInterfaceValues<dim> fe_interface_values;
  };



  struct CopyData
  {
    CopyData(const unsigned int dofs_per_cell)
      : cell_matrix(dofs_per_cell, dofs_per_cell)
      , cell_rhs(dofs_per_cell)
      , local_dof_indices(dofs_per_cell)
    {}


    CopyData(const CopyData &) = default;


    struct FaceData
    {
      FullMatrix<ScalarType>                   cell_matrix;
      std::vector<types::global_dof_index> joint_dof_indices;
    };

    FullMatrix<ScalarType>                   cell_matrix;
    Vector<ScalarType>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<FaceData>                face_data;
  };



  // The more interesting part is where we actually assemble the linear system.
  // Fundamentally, this function has five parts:
  // - The definition of the `cell_worker` "lambda function", a small
  //   function that is defined within the surrounding `assemble_system()`
  //   function and that will be responsible for computing the local
  //   integrals on an individual cell; it will work on a copy of the
  //   `ScratchData` class and put its results into the corresponding
  //   `CopyData` object.
  // - The definition of the `face_worker` lambda function that does
  //   the integration of all terms that live on the interfaces between
  //   cells.
  // - The definition of the `boundary_worker` function that does the
  //   same but for cell faces located on the boundary of the domain.
  // - The definition of the `copier` function that is responsible
  //   for copying all of the data the previous three functions have
  //   put into copy objects for a single cell, into the global matrix
  //   and right hand side.
  //
  // The fifth part is the one where we bring all of this together.
  //
  // Let us go through each of these pieces necessary for the assembly
  // in turns.
  template <int dim>
  void BiharmonicProblem<dim>::assemble_system()
  {
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    // The first piece is the `cell_worker` that does the assembly
    // on the cell interiors. It is a (lambda) function that takes
    // a cell (input), a scratch object, and a copy object (output)
    // as arguments. It looks like the assembly functions of many
    // other of the tutorial programs, or at least the body of the
    // loop over all cells.
    //
    // The terms we integrate here are the cell contribution
    // @f{align*}{
    //    A^K_{ij} = \int_K \nabla^2\varphi_i(x) : \nabla^2\varphi_j(x) dx
    // @f}
    // to the global matrix, and
    // @f{align*}{
    //    f^K_i = \int_K varphi_i(x) f(x) dx
    // @f}
    // to the right hand side vector.
    //
    // We use the same technique as used in the assembly of step-22
    // to accelerate the function: Instead of calling
    // `fe_values.shape_hessian(i, qpoint)` in the innermost loop,
    // we instead create a variable `hessian_i` that evaluates this
    // value once in the loop over `i` and re-use the so-evaluated
    // value in the loop over `j`. For symmetry, we do the same with a
    // variable `hessian_j`, although it is indeed only used once and
    // we could have left the call to `fe_values.shape_hessian(j,qpoint)`
    // in the instruction that computes the scalar product between
    // the two terms.
    auto cell_worker = [&](const Iterator &  cell,
                           ScratchData<dim> &scratch_data,
                           CopyData &        copy_data) {
      copy_data.cell_matrix = 0;
      copy_data.cell_rhs    = 0;

      scratch_data.fe_values.reinit(cell);
      cell->get_dof_indices(copy_data.local_dof_indices);

      const FEValues<dim> &fe_values = scratch_data.fe_values;

      const RightHandSide<dim> right_hand_side (omega);

      const unsigned int dofs_per_cell =
        scratch_data.fe_values.get_fe().dofs_per_cell;

      for (unsigned int qpoint = 0; qpoint < fe_values.n_quadrature_points;
           ++qpoint)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<2,dim> hessian_i = fe_values.shape_hessian(i, qpoint);
              const Tensor<1,dim> grad_i    = fe_values.shape_grad(i, qpoint);
              const double        value_i   = fe_values.shape_value(i, qpoint);
              
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const Tensor<2,dim> hessian_j = fe_values.shape_hessian(j, qpoint);
                  const Tensor<1,dim> grad_j    = fe_values.shape_grad(j, qpoint);
                  const double        value_j   = fe_values.shape_value(j, qpoint);

                  copy_data.cell_matrix(i, j) +=
                    (MaterialParameters::stiffness_D *
                     scalar_product(
                       hessian_i,   // nabla^2 phi_i(x)
                       hessian_j)   // nabla^2 phi_j(x)
                     +
                     MaterialParameters::tension *
                     grad_i *
                     grad_j
                     -
                     omega *
                     omega *
                     MaterialParameters::thickness *
                     MaterialParameters::density *
                     value_i *
                     value_j
                     )
                    * fe_values.JxW(qpoint);                  // dx
                }

              copy_data.cell_rhs(i) +=
                fe_values.shape_value(i, qpoint) * // phi_i(x)
                right_hand_side.value(
                  fe_values.quadrature_point(qpoint)) * // f(x)
                fe_values.JxW(qpoint);                  // dx
            }
        }
    };


    // The next building block is the one that assembles penalty terms on each
    // of the interior faces of the mesh. As described in the documentation of
    // MeshWorker::mesh_loop(), this function receives arguments that denote
    // a cell and its neighboring cell, as well as (for each of the two
    // cells) the face (and potentially sub-face) we have to integrate
    // over. Again, we also get a scratch object, and a copy object
    // for putting the results in.
    //
    // The function has three parts itself. At the top, we initialize
    // the FEInterfaceValues object and create a new `CopyData::FaceData`
    // object to store our input in. This gets pushed to the end of the
    // `copy_data.face_data` variable. We need to do this because
    // the number of faces (or subfaces) over which we integrate for a
    // given cell differs from cell to cell, and the sizes of these
    // matrices also differ, depending on what degrees of freedom
    // are adjacent to the face or subface. As discussed in the documentation
    // of MeshWorker::mesh_loop(), the copy object is reset every time a new
    // cell is visited, so that what we push to the end of
    // `copy_data.face_data()` is really all that the later `copier` function
    // gets to see when it copies the contributions of each cell to the global
    // matrix and right hand side objects.
    auto face_worker = [&](const Iterator &    cell,
                           const unsigned int &f,
                           const unsigned int &sf,
                           const Iterator &    ncell,
                           const unsigned int &nf,
                           const unsigned int &nsf,
                           ScratchData<dim> &  scratch_data,
                           CopyData &          copy_data) {
      FEInterfaceValues<dim> &fe_interface_values =
        scratch_data.fe_interface_values;
      fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);

      copy_data.face_data.emplace_back();
      CopyData::FaceData &copy_data_face = copy_data.face_data.back();

      copy_data_face.joint_dof_indices =
        fe_interface_values.get_interface_dof_indices();

      const unsigned int n_interface_dofs =
        fe_interface_values.n_current_interface_dofs();
      copy_data_face.cell_matrix.reinit(n_interface_dofs, n_interface_dofs);

      // The second part deals with determining what the penalty
      // parameter should be. The simplest formula for this parameter $\gamma$
      // is $\frac{p(p+1)}{h_K}$ where $p$ is the polynomial degree of the
      // finite element used and $h_K$ is the size of cell $K$. But this
      // is not quite so straightforward: If one uses highly stretched cells,
      // then a more involved theory says that $h$ should be replaced be the
      // diameter of cell $K$ normal to the direction of the edge in question.
      // It turns out that there is a function in deal.II for that. Secondly,
      // $h_K$ may be different when viewed from the two different sides of
      // a face.
      //
      // To stay on the safe side, we take the maximum of the two values.
      // We will note that it is possible that this computation has to be
      // further adjusted if one were to use hanging nodes resulting from
      // adaptive mesh refinement.
      const unsigned int p = fe.degree;
      const double       gamma =
        std::max((1.0 * p * (p + 1) /
                  cell->extent_in_direction(
                    GeometryInfo<dim>::unit_normal_direction[f])),
                 (1.0 * p * (p + 1) /
                  ncell->extent_in_direction(
                    GeometryInfo<dim>::unit_normal_direction[nf])));

      // Finally, and as usual, we loop over the quadrature points
      // and indices `i` and `j` to add up the contributions of this
      // face or sub-face. These are then stored in the `copy_data.face_data`
      // object created above. As for the cell worker, we pull the evalation
      // of averages and jumps out of the loops if possible, introducing
      // local variables that store these results. The assembly then only
      // needs to use these local variables in the innermost loop.
      for (unsigned int qpoint = 0;
           qpoint < fe_interface_values.n_quadrature_points;
           ++qpoint)
        {
          const auto &n = fe_interface_values.normal(qpoint);

          for (unsigned int i = 0; i < n_interface_dofs; ++i)
            {
              const double av_hessian_i_dot_n_dot_n
                = (fe_interface_values.average_hessian(i, qpoint) * n * n);
              const double jump_grad_i_dot_n
                = (fe_interface_values.jump_gradient(i, qpoint) * n);
              
              for (unsigned int j = 0; j < n_interface_dofs; ++j)
                {
                  const double av_hessian_j_dot_n_dot_n
                    = (fe_interface_values.average_hessian(j, qpoint) * n * n);
                  const double jump_grad_j_dot_n
                    = (fe_interface_values.jump_gradient(j, qpoint) * n);

              
                  copy_data_face.cell_matrix(i, j) +=
                    MaterialParameters::stiffness_D *                  
                    (-
                     av_hessian_i_dot_n_dot_n // - {grad^2 v n n }
                     * jump_grad_j_dot_n // [grad u n]
                     -
                     av_hessian_j_dot_n_dot_n // - {grad^2 u n n }
                     * jump_grad_i_dot_n // [grad v n]
                     // gamma [grad u n ][grad v n]:
                     +
                     gamma
                     * jump_grad_i_dot_n
                     * jump_grad_j_dot_n
                    )
                    *
                    fe_interface_values.JxW(qpoint); // dx
                }
            }
        }
    };


    // The third piece is to do the same kind of assembly for faces that
    // are at the boundary. The idea is the same as above, of course,
    // with only the difference that there are now penalty terms that
    // also go into the right hand side.
    //
    // As before, the first part of the function simply sets up some
    // helper objects:
    auto boundary_worker = [&](const Iterator &    cell,
                               const unsigned int &face_no,
                               ScratchData<dim> &  scratch_data,
                               CopyData &          copy_data) {
      FEInterfaceValues<dim> &fe_interface_values = scratch_data.fe_interface_values;
      fe_interface_values.reinit(cell, face_no);
      const auto &q_points = fe_interface_values.get_quadrature_points();

      copy_data.face_data.emplace_back();
      CopyData::FaceData &copy_data_face = copy_data.face_data.back();

      const unsigned int n_dofs        = fe_interface_values.n_current_interface_dofs();
      copy_data_face.joint_dof_indices = fe_interface_values.get_interface_dof_indices();

      copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

      const std::vector<double> &        JxW     = fe_interface_values.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = fe_interface_values.get_normal_vectors();


      // Positively, because we now only deal with one cell adjacent to the
      // face (as we are on the boundary), the computation of the penalty
      // factor $\gamma$ is substantially simpler:
      const unsigned int p = fe.degree;
      const double       gamma =
        (1.0 * p * (p + 1) /
         cell->extent_in_direction(
           GeometryInfo<dim>::unit_normal_direction[face_no]));

      // The third piece is the assembly of terms. This is now slightly more
      // involved since these contains both terms for the matrix and for
      // the right hand side. The latter requires us to evaluate the
      //
      for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
        {
          const auto &n = normals[qpoint];

          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const double av_hessian_i_dot_n_dot_n
                = (fe_interface_values.average_hessian(i, qpoint) * n * n);
              const double jump_grad_i_dot_n
                = (fe_interface_values.jump_gradient(i, qpoint) * n);
              
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  const double av_hessian_j_dot_n_dot_n
                    = (fe_interface_values.average_hessian(j, qpoint) * n * n);

                  const double jump_grad_j_dot_n
                    = (fe_interface_values.jump_gradient(j, qpoint) * n);
                  
                  copy_data_face.cell_matrix(i, j) +=
                    MaterialParameters::stiffness_D *
                    (-av_hessian_i_dot_n_dot_n  // - {grad^2 v n n }
                     * jump_grad_j_dot_n        //   * [grad u n]
                     //
                     -av_hessian_j_dot_n_dot_n  // - {grad^2 u n n }
                     * jump_grad_i_dot_n        //   * [grad v n]
                     //
                     +
                     gamma *                    // + gamma
                     jump_grad_i_dot_n          //   * [grad v n]
                     * jump_grad_j_dot_n        //   * [grad u n]
                    ) *
                    JxW[qpoint]; // dx
                }


              // Ordinarily, the rhs vector would contain a term that makes sure the
              // boundary conditions of the form du/dn=h are taken care of. But for the
              // purposes of the current program, h=0 and so the additional term is
              // simply zero. So there is no term of that form we need to add here.              
            }
        }
    };

    // Part 4 was a small function that copies the data produced by the
    // cell, interior, and boundary face assemblers above into the
    // global matrix and right hand side vector. There really is not
    // very much to do here: We distribute the cell matrix and right
    // hand side contributions as we have done in almost all of the
    // other tutorial programs using the constraints objects. We then
    // also have to do the same for the face matrix contributions
    // that have gained content for the faces (interior and boundary)
    // and that the `face_worker` and `boundary_worker` have added
    // to the `copy_data.face_data` array.
    auto copier = [&](const CopyData &copy_data) {
      constraints.distribute_local_to_global(copy_data.cell_matrix,
                                             copy_data.cell_rhs,
                                             copy_data.local_dof_indices,
                                             system_matrix,
                                             system_rhs);

      for (auto &cdf : copy_data.face_data)
        {
          constraints.distribute_local_to_global(cdf.cell_matrix,
                                                 cdf.joint_dof_indices,
                                                 system_matrix);
        }
    };


    // Having set all of this up, what remains is to just create a scratch
    // and copy data object and call the MeshWorker::mesh_loop() function
    // that then goes over all cells and faces, calls the respective workers
    // on them, and then the copier function that puts things into the
    // global matrix and right hand side. As an additional benefit,
    // MeshWorker::mesh_loop() does all of this in parallel, using
    // as many processor cores as your machine happens to have.
    const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;
    ScratchData<dim>   scratch_data(mapping,
                                  fe,
                                  n_gauss_points,
                                  update_values | update_gradients |
                                    update_hessians | update_quadrature_points |
                                    update_JxW_values,
                                  update_values | update_gradients |
                                    update_hessians | update_quadrature_points |
                                    update_JxW_values | update_normal_vectors);
    CopyData           copy_data(dof_handler.get_fe().dofs_per_cell);
    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                            MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);
  }



  // @sect{Solving the linear system and postprocessing}
  //
  // The show is essentially over at this point: The remaining functions are
  // not overly interesting or novel. The first one simply uses a direct
  // solver to solve the linear system (see also step-29):
  template <int dim>
  void BiharmonicProblem<dim>::solve()
  {
    solution = system_rhs;
    
    SparseDirectUMFPACK direct_solver;
    direct_solver.solve(system_matrix, solution);
    constraints.distribute(solution);
  }



  // The next function postprocesses the solution. In the current context,
  // this implies computing the integral over the magnitude of the solution.
  // It will be small in general, except in the vicinity of eigenvalues.
  template <int dim>
  void BiharmonicProblem<dim>::postprocess()
  {
    // Compute the integral of the absolute value of the solution.
    const QGauss<dim>  quadrature_formula(fe.degree + 2);
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values);

    ScalarType integral_solution = 0;
    double integral_p = 0;

    double max_solution = 0;
    double max_p = 0;

    std::vector<ScalarType> function_values_solution(n_q_points);
    std::vector<double> function_values_p(n_q_points);
    for (auto cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(solution, function_values_solution);
        RightHandSide<dim>(omega).value_list (fe_values.get_quadrature_points(),
                                              function_values_p);
            
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            integral_solution += function_values_solution[q_point] *
                                 fe_values.JxW(q_point);
            integral_p        += function_values_p[q_point] *
                                 fe_values.JxW(q_point);

            max_solution = std::max (max_solution,
                                     std::abs(function_values_solution[q_point]));
            max_p        = std::max (max_p,
                                     std::abs(function_values_p[q_point]));
          }
      }

    output_data.normalized_amplitude_integral
      = integral_solution/max_p;
          
    output_data.normalized_maximum_amplitude
      = max_solution/max_p;
  }



  // Equally uninteresting is the function that generates graphical output.
  // It looks exactly like the one in step-6, for example.
  template <int dim>
  void
  BiharmonicProblem<dim>::output_results()
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(fe.degree);

    std::string file_name = "visualization/solution-" +
                            std::to_string(static_cast<unsigned int>(omega/2/numbers::PI)) +
                            ".vtu";
    std::ofstream output_vtu(file_name);
    data_out.write_vtu(output_vtu);

    output_data.visualization_file_name = file_name;
  }



  // The same is true for the `run()` function: Just like in previous
  // programs.
  template <int dim>
  void BiharmonicProblem<dim>::run()
  {
    make_grid();
    setup_system();

    assemble_system();
    solve();

    // Run the generation of graphical output in the background...
    Threads::Task<> output ([this]() { output_results(); });

    // ...while also doing postprocessing.
    postprocess();

    // Then wait for the graphical output task
    output.join();
    

    // Finally, put the result into the output variable that we can
    // read from main(). Make sure that access to the variable is
    // properly guarded across threads.
    std::lock_guard<std::mutex> guard (results_mutex);
    results[omega] = output_data;
  }
} // namespace MembraneOscillation



// @sect3{The main() function}
//
// Finally for the `main()` function. There is, again, not very much to see
// here: It looks like the ones in previous tutorial programs. There
// is a variable that allows selecting the polynomial degree of the element
// we want to use for solving the equation. Because the C0IP formulation
// we use requires the element degree to be at least two, we check with
// an assertion that whatever one sets for the polynomial degree actually
// makes sense.
int main()
{
  try
    {
      using namespace dealii;
      using namespace MembraneOscillation;

      const unsigned int fe_degree = 2;
      Assert(fe_degree >= 2,
             ExcMessage("The C0IP formulation for the biharmonic problem "
                        "only works if one uses elements of polynomial "
                        "degree at least 2."));

      // Get the global set of parameters from an input file
      {
        ParameterHandler prm;
        declare_parameters(prm);
        read_parameters(prm);
      }
      
      // Then sweep over the frequencies.
      Threads::TaskGroup<> tasks;
      unsigned int n_tasks = 0;
      for (double omega=MaterialParameters::min_omega;
           omega<=MaterialParameters::max_omega;
           omega+=(MaterialParameters::max_omega-MaterialParameters::min_omega)/MaterialParameters::n_frequencies,
           ++n_tasks)
        tasks += Threads::new_task ([=]() {
            // The main() function has created tasks for all frequencies
            // provided by the caller, but there is the possibility that a
            // higher instance has decided that the program needs to stop
            // doing what it's doing. Check here, as this is the first
            // non-trivial place one ends up when a task executes, whether we
            // are supposed to actually do anything, or should instead stop
            // working on the frequency this task corresponds to.
            if (check_for_termination_signal() == true)
              {
                std::cout << "Aborting work on omega = " << omega << std::endl;
                return;
              }

            BiharmonicProblem<2> biharmonic_problem(fe_degree, omega);
            biharmonic_problem.run();
          });
      std::cout << "Number of frequencies scheduled: "
                << n_tasks << std::endl;

      tasks.join_all();

      // Output everything previously computed. Make sure that we
      // wait for all threads to release access to the variable. (This
      // is unnecessary here because we have joined all tasks, but
      // it doesn't hurt either.)
      std::lock_guard<std::mutex> guard (results_mutex);
      std::cout << "Number of frequencies computed: "
                << results.size() << std::endl;
      std::ofstream frequency_response ("frequency_response.txt");
      frequency_response << "# Columns are as follows:\n"
                         << "# 1: Frequency [Hz]\n"
                         << "# 2: Real part of the volume displacement divided by the amplitude of pressure [m^3/Pa]\n"
                         << "# 3: Imaginary part of the same\n"
                         << "# 4: Real part of the impedance [Pa.s/m^3]\n"
                         << "# 5: Imaginary part of the same\n"
                         << "# 6: Absolute value of maximal displacement divided by amplitude of pressure [m/Pa]\n"
                         << "# 5: File name for graphical output of the displacement visualization.\n";
      for (auto result : results)
        {
          const auto omega = result.first;
          const auto impedance = 1./result.second.normalized_amplitude_integral
                                 /omega/std::complex<double>(0.,1.);
          
          frequency_response << omega/2/numbers::PI << ' '
                             << std::real(result.second.normalized_amplitude_integral) << ' '
                             << std::imag(result.second.normalized_amplitude_integral) << ' '
                             << std::real(impedance) << ' '
                             << std::imag(impedance) << ' '
                             << result.second.normalized_maximum_amplitude << ' '
                             << '"' << result.second.visualization_file_name << '"'
                             << std::endl;
        }
      

      // Whether or not a termination signal has been sent, try to
      // remove the file that indicates this signal. That's because if
      // we don't do that, the next call to this program won't produce
      // anything at all.
      std::remove ("termination_signal");
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
