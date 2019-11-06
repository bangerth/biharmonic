/*

  biharmonic using FEInterfaceValues


TODO:
- bug: acess invalid matrix entry
- missing boundary terms
- calculate penalty terms correctly
*/
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>

//
#include <deal.II/grid/grid_out.h>
//


#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>


// The include files for using the MeshWorker framework
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/mesh_loop.h>

// The include file for local integrators associated with the Laplacian
#include <deal.II/integrators/laplace.h>
//#include <deal.II/integrators/biharmonic.h>



#include "biharmonic.h"

#include <deal.II/base/function_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>


#include <fstream>
#include <iostream>
#include <list>

#ifndef PI
#  define PI 3.14159265358979323846
#endif

namespace StepBiharmonic
{
  using namespace dealii;
  using ConstraintMatrix = AffineConstraints<double>;

  const unsigned int MaxCycle = 6;



  namespace simple
  {
    template <int dim>
    class Solution : public Function<dim>
    {
    public:
      Solution()
        : Function<dim>()
      {}
      virtual double value(const Point<dim> & p,
                           const unsigned int /*component*/ = 0) const
      {
        return sin(PI * p[0]) * sin(PI * p[1]);
      }

      virtual Tensor<1, dim> gradient(const Point<dim> & p,
                                      const unsigned int /*component*/ = 0) const
      {
        Tensor<1, dim> r;
        r[0] = PI * cos(PI * p[0]) * sin(PI * p[1]);
        r[1] = PI * cos(PI * p[1]) * sin(PI * p[0]);
        return r;
      }

      virtual void hessian_list(const std::vector<Point<dim>> &       points,
                                std::vector<SymmetricTensor<2, dim>> &hessians,
                                const unsigned int /*component*/ = 0) const
      {
        for (unsigned i = 0; i < points.size(); ++i)
          {
            const double x = points[i][0];
            const double y = points[i][1];

            hessians[i][0][0] = -PI * PI * sin(PI * x) * sin(PI * y);
            hessians[i][0][1] = PI * PI * cos(PI * x) * cos(PI * y);
            hessians[i][1][1] = -PI * PI * sin(PI * x) * sin(PI * y);
          }
      }
    };



    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
      RightHandSide()
        : Function<dim>()
      {}

      virtual double value(const Point<dim> & p,
                           const unsigned int /*component*/ = 0) const

      {
        return 4 * std::pow(PI, 4.0) * sin(PI * p[0]) * sin(PI * p[1]);
      }
    };
  } // namespace simple

  namespace sin4
  {
    template <int dim>
    class Solution : public Function<dim>
    {
    public:
      Solution()
        : Function<dim>()
      {}
      virtual double         value(const Point<dim> & p,
                                   const unsigned int component = 0) const;
      virtual Tensor<1, dim> gradient(const Point<dim> & p,
                                      const unsigned int component = 0) const;
      virtual void           hessian_list(const std::vector<Point<dim>> &       points,
                                          std::vector<SymmetricTensor<2, dim>> &hessians,
                                          const unsigned int component = 0) const;
    };


    template <int dim>
    double Solution<dim>::value(const Point<dim> &p, const unsigned int) const
    {
      return std::sin(PI * p(0)) * std::sin(PI * p(0)) * std::sin(PI * p(1)) *
             std::sin(PI * p(1));
    }

    template <int dim>
    Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p,
                                           const unsigned int) const
    {
      Tensor<1, dim> return_value;

      return_value[0] = 2 * PI * std::cos(PI * p(0)) * std::sin(PI * p(0)) *
                        std::sin(PI * p(1)) * std::sin(PI * p(1));
      return_value[1] = 2 * PI * std::cos(PI * p(1)) * std::sin(PI * p(0)) *
                        std::sin(PI * p(0)) * std::sin(PI * p(1));
      return return_value;
    }

    template <int dim>
    void
    Solution<dim>::hessian_list(const std::vector<Point<dim>> &       points,
                                std::vector<SymmetricTensor<2, dim>> &hessians,
                                const unsigned int) const
    {
      Tensor<1, dim> p;
      for (unsigned i = 0; i < points.size(); ++i)
        {
          for (unsigned int d = 0; d < dim; ++d)
            {
              p[d] = points[i][d];
            } // d-loop
          for (unsigned int d = 0; d < dim; ++d)
            {
              hessians[i][d][d] =
                2 * PI * PI * std::cos(PI * p[d]) * std::cos(PI * p[d]) *
                  std::sin(PI * p[(d + 1) % dim]) *
                  std::sin(PI * p[(d + 1) % dim]) -
                2 * PI * PI * std::sin(PI * p[d]) * std::sin(PI * p[d]) *
                  std::sin(PI * p[(d + 1) % dim]) *
                  std::sin(PI * p[(d + 1) % dim]);
              hessians[i][d][(d + 1) % dim] =
                4 * PI * PI * std::cos(PI * p[d]) *
                std::cos(PI * p[(d + 1) % dim]) * std::sin(PI * p[d]) *
                std::sin(PI * p[(d + 1) % dim]);
              hessians[i][(d + 1) % dim][d] = hessians[i][d][(d + 1) % dim];
            }
        }
    }

    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
      RightHandSide()
        : Function<dim>()
      {}

      virtual double value(const Point<dim> & p,
                           const unsigned int component = 0) const;
    };

    template <int dim>
    double RightHandSide<dim>::value(const Point<dim> & p,
                                     const unsigned int component) const
    {
      Assert(component == 0, ExcNotImplemented());

      return (8 * std::pow(PI, 4) *
              (8 * std::sin(PI * p(0)) * std::sin(PI * p(0)) *
                 std::sin(PI * p(1)) * std::sin(PI * p(1)) -
               3 * std::sin(PI * p(0)) * std::sin(PI * p(0)) -
               3 * std::sin(PI * p(1)) * std::sin(PI * p(1)) + 1));
    }
  } // namespace sin4


  using namespace simple;
  // using namespace sin4;



  template <int dim>
  class ErrorIntegrator : public MeshWorker::LocalIntegrator<dim>
  {
  public:
    void cell(MeshWorker::DoFInfo<dim> &                 dinfo,
              typename MeshWorker::IntegrationInfo<dim> &info) const;
    void boundary(MeshWorker::DoFInfo<dim> &                 dinfo,
                  typename MeshWorker::IntegrationInfo<dim> &info) const;
    void face(MeshWorker::DoFInfo<dim> &                 dinfo1,
              MeshWorker::DoFInfo<dim> &                 dinfo2,
              typename MeshWorker::IntegrationInfo<dim> &info1,
              typename MeshWorker::IntegrationInfo<dim> &info2) const;
  };
  template <int dim>
  void ErrorIntegrator<dim>::cell(
    MeshWorker::DoFInfo<dim> &                 dinfo,
    typename MeshWorker::IntegrationInfo<dim> &info) const
  {
    const FEValuesBase<dim> &   fe = info.fe_values();
    std::vector<double>         exact_values(fe.n_quadrature_points);
    std::vector<Tensor<1, dim>> exact_gradients(fe.n_quadrature_points);
    std::vector<SymmetricTensor<2, dim>> exact_hessians(fe.n_quadrature_points);

    const Solution<dim> extsol;


    extsol.value_list(fe.get_quadrature_points(), exact_values);
    extsol.gradient_list(fe.get_quadrature_points(), exact_gradients);
    extsol.hessian_list(fe.get_quadrature_points(), exact_hessians);

    const std::vector<double> &        uh   = info.values[0][0];
    const std::vector<Tensor<1, dim>> &Duh  = info.gradients[0][0];
    const std::vector<Tensor<2, dim>> &DDuh = info.hessians[0][0];

    for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
      {
        exact_hessians[k][0][0] -= DDuh[k][0][0];
        exact_hessians[k][1][1] -= DDuh[k][1][1];
        exact_hessians[k][1][0] -= DDuh[k][1][0];
        // exact_hessians[k][0][1] -=DDuh[k][0][1]; exact_hessians is a
        // symmetric tensor, so don't edit both

        double sum = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          {
            const double diff_grad  = exact_gradients[k][d] - Duh[k][d];
            const double diff_hess1 = exact_hessians[k][d][d];
            const double diff_hess2 = exact_hessians[k][d][(d + 1) % dim];
            sum += (0 * diff_grad * diff_grad + diff_hess1 * diff_hess1 +
                    diff_hess2 * diff_hess2);
          }
        dinfo.value(0) += sum * fe.JxW(k);
        const double diff = exact_values[k] - uh[k];
        dinfo.value(1) += diff * diff * fe.JxW(k);
      }
    dinfo.value(0) = std::sqrt(dinfo.value(0));
    dinfo.value(1) = std::sqrt(dinfo.value(1));
  }


  template <int dim>
  void ErrorIntegrator<dim>::boundary(
    MeshWorker::DoFInfo<dim> &                 dinfo,
    typename MeshWorker::IntegrationInfo<dim> &info) const
  {
    const Solution<dim> extsol;


    const FEValuesBase<dim> &   fe = info.fe_values();
    std::vector<Tensor<1, dim>> exact_gradients(fe.n_quadrature_points);
    extsol.gradient_list(fe.get_quadrature_points(), exact_gradients);

    const std::vector<Tensor<1, dim>> &Duh = info.gradients[0][0];
    const unsigned int                 deg = fe.get_fe().tensor_degree();
    const double                       penalty =
      dealii::LocalIntegrators::Biharmonic::compute_penalty(dinfo,
                                                            dinfo,
                                                            deg,
                                                            deg);

    for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
      {
        const double diff = (exact_gradients[k] - Duh[k]) * fe.normal_vector(k);
        dinfo.value(0) += penalty * diff * diff * fe.JxW(k);
      }
    dinfo.value(0) = std::sqrt(dinfo.value(0));
  }

  template <int dim>
  void ErrorIntegrator<dim>::face(
    MeshWorker::DoFInfo<dim> &                 dinfo1,
    MeshWorker::DoFInfo<dim> &                 dinfo2,
    typename MeshWorker::IntegrationInfo<dim> &info1,
    typename MeshWorker::IntegrationInfo<dim> &info2) const
  {
    const FEValuesBase<dim> &          fe   = info1.fe_values();
    const std::vector<Tensor<1, dim>> &Duh1 = info1.gradients[0][0];
    const std::vector<Tensor<1, dim>> &Duh2 = info2.gradients[0][0];
    const unsigned int                 deg  = fe.get_fe().tensor_degree();
    const double                       penalty =
      dealii::LocalIntegrators::Biharmonic::compute_penalty(dinfo1,
                                                            dinfo2,
                                                            deg,
                                                            deg);

    for (unsigned k = 0; k < fe.n_quadrature_points; ++k)
      {
        double diff = (Duh1[k] - Duh2[k]) * fe.normal_vector(k);
        dinfo1.value(0) += (penalty * diff * diff) * fe.JxW(k);
      }
    dinfo1.value(0) = std::sqrt(dinfo1.value(0));
    dinfo2.value(0) = dinfo1.value(0);
  }


  /*************************************************************/
  // @sect3{The main class}
  template <int dim>
  class BiharmonicProblem
  {
  public:
    typedef MeshWorker::IntegrationInfo<dim> CellInfo;

    BiharmonicProblem(const FiniteElement<dim> &fe);

    void run();

  private:
    void   make_grid();
    void   setup_system();
    void   assemble_system();
    void   solve();
    void   error();
    void   output_results(const unsigned int iteration) const;

    Triangulation<dim>        triangulation;
    const MappingQ<dim>       mapping;
    const FiniteElement<dim> &fe;
    DoFHandler<dim>           dof_handler;
    ConstraintMatrix          constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix, complete_system_matrix;


    Vector<double>      solution;
    Vector<double>      system_rhs;
    Vector<double>      complete_system_rhs;
    Vector<double>      diagonal_of_mass_matrix;
  };

  template <int dim>
  BiharmonicProblem<dim>::BiharmonicProblem(const FiniteElement<dim> &fe)
    : mapping(/*2*/ 1)
    , fe(fe)
    , dof_handler(triangulation)
  {}



  template <int dim>
  void BiharmonicProblem<dim>::make_grid()
  {
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    triangulation.refine_global(1);

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: " << triangulation.n_cells()
              << std::endl;
  }



  template <int dim>
  void BiharmonicProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Solution<dim>(),
                                             constraints);

    constraints.close();


    DynamicSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         c_sparsity,
                                         constraints,
                                         true);
    sparsity_pattern.copy_from(c_sparsity);

    system_matrix.reinit(sparsity_pattern);
    complete_system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    complete_system_rhs.reinit(dof_handler.n_dofs());
  }



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
      , fe_interface_values(
          scratch_data.fe_values
            .get_mapping(), // TODO: implement for fe_interface_values
          scratch_data.fe_values.get_fe(),
          scratch_data.fe_interface_values.get_quadrature(),
          scratch_data.fe_interface_values.get_update_flags())
    {}

    FEValues<dim>          fe_values;
    FEInterfaceValues<dim> fe_interface_values;
  };



  struct CopyDataFace
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  };



  struct CopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace>            face_data;

    template <class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };



  template <int dim>
  void BiharmonicProblem<dim>::assemble_system()
  {
    typedef decltype(dof_handler.begin_active()) Iterator;
    const RightHandSide<dim>                     right_hand_side;

    auto cell_worker = [&](const Iterator &  cell,
                           ScratchData<dim> &scratch_data,
                           CopyData &        copy_data) {
      const unsigned int n_dofs = scratch_data.fe_values.get_fe().dofs_per_cell;
      copy_data.reinit(cell, n_dofs);
      scratch_data.fe_values.reinit(cell);

      const auto &q_points = scratch_data.fe_values.get_quadrature_points();

      const FEValues<dim> &      fe_v = scratch_data.fe_values;
      const std::vector<double> &JxW  = fe_v.get_JxW_values();

      // scalar_product(fe.shape_hessian_component(j,k,d),
      // fe.shape_hessian_component(i,k,d));
      const double nu = 1.0;

      for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
        {
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  // \int_Z \nu \nabla^2 u \cdot \nabla^2 v \, dx.
                  copy_data.cell_matrix(i, j) +=
                    nu *
                    scalar_product(fe_v.shape_hessian(i, point),
                                   fe_v.shape_hessian(j, point)) *
                    JxW[point]; // dx
                }


              copy_data.cell_rhs(i) += fe_v.shape_value(i, point) *
                                       right_hand_side.value(q_points[point]) *
                                       JxW[point]; // dx
            }
        }
    };


    auto face_worker = [&](const Iterator &    cell,
                           const unsigned int &f,
                           const unsigned int &sf,
                           const Iterator &    ncell,
                           const unsigned int &nf,
                           const unsigned int &nsf,
                           ScratchData<dim> &  scratch_data,
                           CopyData &          copy_data) {
      FEInterfaceValues<dim> &fe_i = scratch_data.fe_interface_values;
      fe_i.reinit(cell, f, sf, ncell, nf, nsf);
      const auto &q_points = fe_i.get_quadrature_points();

      copy_data.face_data.emplace_back();
      CopyDataFace &copy_data_face = copy_data.face_data.back();

      const unsigned int n_dofs        = fe_i.n_current_interface_dofs();
      copy_data_face.joint_dof_indices = fe_i.get_interface_dof_indices();

      copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

      const std::vector<double> &        JxW     = fe_i.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = fe_i.get_normal_vectors();

      // eta = 1/2 + 2C_2
      // gamma = eta/|e|

      double gamma = 1.0; // TODO:

      {
        int                degree = fe.tensor_degree();
        const unsigned int normal1 =
          GeometryInfo<dim>::unit_normal_direction[f];
        const unsigned int normal2 =
          GeometryInfo<dim>::unit_normal_direction[nf];
        const unsigned int deg1sq =
          degree * (degree + 1); //(deg1 == 0) ? 1 : deg1 * (deg1+1);
        const unsigned int deg2sq =
          degree * (degree + 1); //(deg2 == 0) ? 1 : deg2 * (deg2+1);

        double penalty1 = deg1sq / cell->extent_in_direction(normal1);
        double penalty2 = deg2sq / ncell->extent_in_direction(normal2);
        if (cell->has_children() ^ ncell->has_children())
          {
            penalty1 *= 8;
          }
        gamma = 0.5 * (penalty1 + penalty2);
      }


        for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
        {
          // \int_F {grad^2 u n n } [grad v n]
          //   - {grad^2 v n n } [grad u n]
          //   -  gamma [grad u n ][grad v n]
          const auto &n = normals[qpoint];

          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                Assert((fe_i.average_hessian(i, qpoint) * n * n) ==
                         contract3(n, fe_i.average_hessian(i, qpoint), n),
                       ExcInternalError());

                Assert((fe_i.jump_gradient(j, qpoint) * n) ==
                         (n * fe_i.jump_gradient(j, qpoint)),
                       ExcInternalError());


                copy_data_face.cell_matrix(i, j) +=
                  (
                    // {grad^2 v n n } [grad u n]
                    -(fe_i.average_hessian(i, qpoint) * n * n)
                      //                    -(trace(fe_i.average_hessian(i,qpoint)))
                      * (fe_i.jump_gradient(j, qpoint) * n) //
                    // {grad^2 u n n } [grad v n]
                    -
                    (fe_i.average_hessian(j, qpoint) * n * n)
                      //                    -
                      //                    (trace(fe_i.average_hessian(j,qpoint)))
                      * (fe_i.jump_gradient(i, qpoint) * n) //
                    // gamma [grad u n ][grad v n]
                    + gamma * (fe_i.jump_gradient(i, qpoint) * n) *
                        (fe_i.jump_gradient(j, qpoint) * n)) *
                  JxW[qpoint]; // dx
              }
        }
    };


    auto boundary_worker = [&](const Iterator &    cell,
                               const unsigned int &face_no,
                               ScratchData<dim> &  scratch_data,
                               CopyData &          copy_data) {
      // return;
      FEInterfaceValues<dim> &fe_i = scratch_data.fe_interface_values;
      fe_i.reinit(cell, face_no);
      const auto &q_points = fe_i.get_quadrature_points();

      copy_data.face_data.emplace_back();
      CopyDataFace &copy_data_face = copy_data.face_data.back();

      const unsigned int n_dofs        = fe_i.n_current_interface_dofs();
      copy_data_face.joint_dof_indices = fe_i.get_interface_dof_indices();

      copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

      const std::vector<double> &        JxW     = fe_i.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = fe_i.get_normal_vectors();


      const Solution<dim>         extsol;
      std::vector<Tensor<1, dim>> exact_gradients(q_points.size());
      extsol.gradient_list(q_points, exact_gradients);


      // eta = 1/2 + 2C_2
      // gamma = eta/|e|

      double gamma = 1.0;

      {
        int                degree = fe.tensor_degree();
        const unsigned int normal1 =
          GeometryInfo<dim>::unit_normal_direction[face_no];
        const unsigned int deg1sq =
          degree * (degree + 1); //(deg1 == 0) ? 1 : deg1 * (deg1+1);

        gamma = deg1sq / cell->extent_in_direction(normal1);
        //      gamma = 0.5*(penalty1 + penalty2);
      }

      for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
        {
          const auto &n = normals[qpoint];

          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              for (unsigned int j = 0; j < n_dofs; ++j)
                copy_data_face.cell_matrix(i, j) +=
                  (
                    // - {grad^2 v n n } [grad u n]
                    -(fe_i.average_hessian(i, qpoint) * n * n) *
                      (fe_i.jump_gradient(j, qpoint) * n) //
                    // - {grad^2 u n n } [grad v n]
                    - (fe_i.average_hessian(j, qpoint) * n * n) *
                        (fe_i.jump_gradient(i, qpoint) * n) //
                    // gamma [grad u n ][grad v n]
                    + 2.0 * gamma * (fe_i.jump_gradient(i, qpoint) * n) *
                        (fe_i.jump_gradient(j, qpoint) * n)) *
                  JxW[qpoint]; // dx


              copy_data.cell_rhs(i) +=
                (-(fe_i.average_hessian(i, qpoint) * n * n) *
                   (exact_gradients[qpoint] * n) //
                 + 2.0 * gamma *                 // why 2?
                     (fe_i.jump_gradient(i, qpoint) * n) *
                     (exact_gradients[qpoint] * n) //
                 ) *
                JxW[qpoint]; // dx
            }
        }
    };

    auto copier = [&](const CopyData &c) {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             system_matrix,
                                             system_rhs);

      for (auto &cdf : c.face_data)
        {
          constraints.distribute_local_to_global(cdf.cell_matrix,
                                                 cdf.joint_dof_indices,
                                                 system_matrix);
        }
    };

    const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;

    ScratchData<dim> scratch_data(mapping,
                                  fe,
                                  n_gauss_points,
                                  update_values | update_gradients |
                                    update_hessians | update_quadrature_points |
                                    update_JxW_values,
                                  update_values | update_gradients |
                                    update_hessians | update_quadrature_points |
                                    update_JxW_values | update_normal_vectors);
    CopyData         copy_data;
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


  template <int dim>
  void BiharmonicProblem<dim>::solve()
  {
    std::cout << "   Solving system..." << std::endl;

    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult(solution, system_rhs);
    constraints.distribute(solution);
  }



  template <int dim>
  void BiharmonicProblem<dim>::error()
  {
    BlockVector<double> errors(2);
    errors.block(0).reinit(triangulation.n_active_cells());
    errors.block(1).reinit(triangulation.n_active_cells());
    unsigned int i = 0;
    for (typename Triangulation<dim>::active_cell_iterator cell =
           triangulation.begin_active();
         cell != triangulation.end();
         ++cell, ++i)
      cell->set_user_index(i);
    MeshWorker::IntegrationInfoBox<dim> info_box;

    const unsigned int n_gauss_points =
      dof_handler.get_fe().tensor_degree() + 1;

    AnyData solution_data;
    solution_data.add(&solution, "solution");
    info_box.cell_selector.add("solution", true, true, true);
    info_box.boundary_selector.add("solution", true, true, false);
    info_box.face_selector.add("solution", true, true, false);

    UpdateFlags update_flags = update_values | update_quadrature_points |
                               update_gradients | update_hessians;
    info_box.add_update_flags_all(update_flags);
    info_box.initialize_gauss_quadrature(n_gauss_points,
                                         n_gauss_points + 1,
                                         n_gauss_points);
    info_box.initialize(fe, mapping, solution_data, solution);
    MeshWorker::DoFInfo<dim>                     dof_info(dof_handler);
    MeshWorker::Assembler::CellsAndFaces<double> assembler;
    AnyData                                      out_data;
    BlockVector<double> *                        est = &errors;
    out_data.add(est, "cells");
    assembler.initialize(out_data, false);
    ErrorIntegrator<dim> integrator;
    MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(),
                                           dof_handler.end(),
                                           dof_info,
                                           info_box,
                                           integrator,
                                           assembler);
    std::cout << "   energy-error: " << errors.block(0).l2_norm() << std::endl;
    std::cout << "   L2-error: " << errors.block(1).l2_norm() << std::endl;

    {
      Vector<float> norm_per_cell(triangulation.n_active_cells());
      VectorTools::integrate_difference(dof_handler,
                                        solution,
                                        Solution<dim>(),
                                        norm_per_cell,
                                        QGauss<dim>(n_gauss_points + 1),
                                        VectorTools::L2_norm);
      const double solution_norm =
        VectorTools::compute_global_error(triangulation,
                                          norm_per_cell,
                                          VectorTools::L2_norm);
      std::cout << "   l2 timo " << solution_norm << std::endl;

      {
        const QGauss<dim> quadrature_formula(fe.degree + 2);
        Solution<dim>     exact_solution;
        Vector<double>    error_per_cell(triangulation.n_active_cells());

        FEValues<dim> fe_values( // mappingfe,
          fe,
          quadrature_formula,
          update_values | update_hessians | update_quadrature_points |
            update_JxW_values);

        FEValuesExtractors::Scalar scalar(0);
        const unsigned int         n_q_points = quadrature_formula.size();

        std::vector<SymmetricTensor<2, dim>> exact_hessians(n_q_points);
        std::vector<Tensor<2, dim>>          hessians(n_q_points);
        unsigned int                         id = 0;
        for (auto cell : dof_handler.active_cell_iterators())
          {
            fe_values.reinit(cell);
            fe_values[scalar].get_function_hessians(solution, hessians);
            exact_solution.hessian_list(fe_values.get_quadrature_points(),
                                        exact_hessians);

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                error_per_cell[id] +=
                  ((exact_hessians[q_point] - hessians[q_point]).norm() *
                   fe_values.JxW(q_point));
              }
            ++id;
          }
        const double h2_semi = std::sqrt(error_per_cell.l2_norm());
        std::cout << "   h2semi timo " << h2_semi << std::endl;
      }
      //      VectorTools::integrate_difference(dof_handler,
      //                                        solution,
      //                                        Solution<dim>(),
      //                                        norm_per_cell,
      //                                        QGauss<dim>(n_gauss_points+1),
      //                                        VectorTools::H2);
      //      const double solution_normh2 =
      //        VectorTools::compute_global_error(triangulation,
      //                                          norm_per_cell,
      //                                          VectorTools::H2_se);
      //      std::cout << "h2 timo " << solution_normh2 << std::endl;
    }
  }


  template <int dim>
  void
  BiharmonicProblem<dim>::output_results(const unsigned int iteration) const
  {
    std::cout << "   Writing graphical output..." << std::endl;

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u");
    Vector<double>      exact  = solution;
    unsigned int        degree = fe.tensor_degree();
    const Solution<dim> exact_solution;
    VectorTools::project(mapping,
                         dof_handler,
                         constraints,
                         QGauss<dim>(degree + 1),
                         exact_solution,
                         exact);
    data_out.add_data_vector(exact, "exact");

    data_out.build_patches();

    std::ofstream output_vtk(
      ("output_" + Utilities::int_to_string(iteration, 6) + ".vtk")
        .c_str());
    data_out.write_vtk(output_vtk);
  }




  template <int dim>
  void BiharmonicProblem<dim>::run()
  {
    GridGenerator::hyper_cube(triangulation, 0, 1);

    for (unsigned int cycle = 0; cycle < MaxCycle; ++cycle)
      {
        std::cout << "Cycle: " << cycle << " of " << MaxCycle << std::endl;



        triangulation.refine_global(1);
        setup_system();

        assemble_system();
        solve();

        output_results(cycle);

        error();
        std::cout << std::endl;
      }
  }
} // namespace StepBiharmonic



int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace StepBiharmonic;
      using namespace LocalIntegrators;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

      int degree = 2; // minimum degree 2
      if (argc > 1)
        degree = Utilities::string_to_int(argv[1]);
      FE_Q<2> fe1(degree);
      std::cout << "FE: " << fe1.get_name() << std::endl;
      BiharmonicProblem<2> my_bi(fe1);
      my_bi.run();
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
