//---------------------------------------------------------------------------
//    $Id: biharmonic.h 25370 2012-04-02 20:53:57Z kanschat $
//
//    Copyright (C) 2010, 2011, 2012 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//---------------------------------------------------------------------------
#ifndef __deal2__integrators_biharmonic_h
#define __deal2__integrators_biharmonic_h


#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/meshworker/dof_info.h>

DEAL_II_NAMESPACE_OPEN

namespace LocalIntegrators
{
/**
 * @brief Local integrators related to the Laplacian and its DG formulations
 *
 * @ingroup Integrators
 * @author Guido Kanschat
 * @date 2010
 */
  namespace Biharmonic
  {
/* Calculation of the second order interior face residual 
*/
template<int dim>

Tensor<1, dim>
second_partial_n(const Tensor<1,dim>& normal,const Tensor<2,dim>& D)
{
  //Tensor<1,dim> result ;
  //contract(result, normal,D);
  //return result ;
 return contract<1,0>(D,normal);
}


/**
 * Biharmonic operator in weak form, namely on the cell <i>Z</i> the matrix
 * \f[
 * \int_Z \nu \nabla^2 u \cdot \nabla^2 v \, dx.
 * \f]
 *
 * The FiniteElement in <tt>fe</tt> may be scalar or vector valued. In
 * the latter case, the Laplacian is applied to each component
 * separately.
 *
 * @ingroup Integrators
 * @author Natasha Sharma
 * @date 2012
 */
    template<int dim>
    void cell_matrix (
      FullMatrix<double>& M,
      const FEValuesBase<dim>& fe,
      const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_components = fe.get_fe().n_components();
      
      AssertDimension(M.m(), n_dofs);
      AssertDimension(M.n(), n_dofs);
      
      for (unsigned k=0;k<fe.n_quadrature_points;++k)
        {
          const double dx = fe.JxW(k) * factor;
          for (unsigned i=0;i<n_dofs;++i)
	    for (unsigned j=0;j<n_dofs;++j)
	      for (unsigned int d=0;d<n_components;++d)
		M(i,j) += dx *
			  //double_contract(fe.shape_hessian_component(j,k,d), fe.shape_hessian_component(i,k,d));
			  scalar_product(fe.shape_hessian_component(j,k,d), fe.shape_hessian_component(i,k,d));
        }
    }

/**
 * The matrix associated with the bilinear form
 * \f[
 * \int_Z \nu \Delta u \cdot \Delta v \, dx.
 * \f] 
 */
    template<int dim>
    void delta_delta_matrix (
      FullMatrix<double>& M,
      const FEValuesBase<dim>& fe,
      const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_components = fe.get_fe().n_components();
      
      AssertDimension(M.m(), n_dofs);
      AssertDimension(M.n(), n_dofs);
      
      for (unsigned k=0;k<fe.n_quadrature_points;++k)
        {
          const double dx = fe.JxW(k) * factor;
          for (unsigned i=0;i<n_dofs;++i)
            {
              for (unsigned j=0;j<n_dofs;++j)
                for (unsigned int c=0;c<n_components;++c)
                  M(i,j) += dx
			    * trace(fe.shape_hessian_component(j,k,c))
			    * trace(fe.shape_hessian_component(i,k,c));
            }
        }
    }
/**
 * Weak boundary condition of Nitsche type for the biharmonic equation, namely on the face <i>F</i> the matrix
 * @f[
 * \int_F \Bigl(\gamma u v - \partial_n u v - u \partial_n v\Bigr)\;ds.
 * @f]
 *
 * Here, $\gamma$ is the <tt>penalty</tt> parameter suitably computed
 * with compute_penalty().
 *
 * @ingroup Integrators
 * @author Natasha Sharma
 * @date 2012
 */
    template <int dim>
    void weak_boundary_matrix (
      FullMatrix<double>& M,
      const FEValuesBase<dim>& fe,
      double penalty,
      double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comp = fe.get_fe().n_components();

      AssertDimension (M.m(), n_dofs);
      AssertDimension (M.n(), n_dofs);
      
      for (unsigned k=0;k<fe.n_quadrature_points;++k)
        {
          const double dx = fe.JxW(k) * factor;
          const Tensor<1, dim>& n = fe.normal_vector(k);
          for (unsigned i=0;i<n_dofs;++i)
            for (unsigned j=0;j<n_dofs;++j)
              for (unsigned int d=0;d<n_comp;++d)
                M(i,j) += dx *
		  (2.*( n * fe.shape_grad_component(i,k,d)) * penalty * (n * fe.shape_grad_component(j,k,d))
		   - contract3(n , fe.shape_hessian_component(i,k,d), n) * (n * fe.shape_grad_component(j,k,d))
		   - contract3(n , fe.shape_hessian_component(j,k,d), n) * (n * fe.shape_grad_component(i,k,d)));
        }
    }


/**
 * Flux for the interior penalty method for the Laplacian, namely on
 * the face <i>F</i> the matrices associated with the bilinear form
 * @f[
 * \int_F \Bigl( \gamma [u][v] - \{\nabla u\}[v\mathbf n] - [u\mathbf
 * n]\{\nabla v\} \Bigr) \; ds.
 * @f]
 *
 * The penalty parameter should always be the mean value of the
 * penalties needed for stability on each side. In the case of
 * constant coefficients, it can be computed using compute_penalty().
 *
 * If <tt>factor2</tt> is missing or negative, the factor is assumed
 * the same on both sides. If factors differ, note that the penalty
 * parameter has to be computed accordingly.
 *
 * @ingroup Integrators
 * @author Natasha Sharma
 * @date 2012
 */
    template <int dim>
    void ip_matrix (
      FullMatrix<double>& M11,
      FullMatrix<double>& M12,
      FullMatrix<double>& M21,
      FullMatrix<double>& M22,
      const FEValuesBase<dim>& fe1,
      const FEValuesBase<dim>& fe2,
      double penalty,
      double factor1 = 1.,
      double factor2 = -1.)
    {
      const unsigned int n_dofs = fe1.dofs_per_cell;
      AssertDimension(M11.n(), n_dofs);
      AssertDimension(M11.m(), n_dofs);
      AssertDimension(M12.n(), n_dofs);
      AssertDimension(M12.m(), n_dofs);
      AssertDimension(M21.n(), n_dofs);
      AssertDimension(M21.m(), n_dofs);
      AssertDimension(M22.n(), n_dofs);
      AssertDimension(M22.m(), n_dofs);

      //const double fi = factor1;
      const double fe = (factor2 < 0) ? factor1 : factor2;

      for (unsigned k=0;k<fe1.n_quadrature_points;++k)
	{
	  const double dx = fe1.JxW(k);
	  const Tensor<1,dim>& n = fe1.normal_vector(k);
	  for (unsigned int d=0;d<fe1.get_fe().n_components();++d)
	    {
	      for (unsigned i=0;i<n_dofs;++i)
		{
		  for (unsigned j=0;j<n_dofs;++j)
		    {
		    
		      const double dnvi   = n * fe1.shape_grad_component(i,k,d);
		      const double dnui   = n * fe1.shape_grad_component(j,k,d);
		      const double ddnvi = contract3(n, fe1.shape_hessian_component(i,k,d),n);
		      const double ddnui = contract3(n, fe1.shape_hessian_component(j,k,d),n);

		      const double dnve = -(n * fe2.shape_grad_component(i,k,d));
		      const double dnue = -(n * fe2.shape_grad_component(j,k,d));
		      const double ddnve = contract3(n, fe2.shape_hessian_component(i,k,d), n);
		      const double ddnue = contract3(n, fe2.shape_hessian_component(j,k,d), n);
                
                M11(i,j) += dx*(-.5*ddnvi*dnui-.5*ddnui*dnvi+penalty*dnui*dnvi);
                M12(i,j) += dx*(-.5*ddnvi*dnue-.5*ddnue*dnvi+penalty*dnue*dnvi);
                M21(i,j) += dx*(-.5*fe*ddnve*dnui-.5*ddnui*dnve+penalty*dnui*dnve);
                M22(i,j) += dx*(-.5*fe*ddnve*dnue-.5*ddnue*dnve+penalty*dnue*dnve);
		      
//		      M11(i,j) += dx*(-.5*fi*ddnvi*dnui-.5*fi*ddnui*dnvi+penalty*dnui*dnvi);
//		      M12(i,j) += dx*(-.5*fi*ddnvi*dnue-.5*fe*ddnue*dnvi+penalty*dnue*dnvi);
//		      M21(i,j) += dx*(-.5*fe*ddnve*dnui-.5*fi*ddnui*dnve+penalty*dnui*dnve);
//		      M22(i,j) += dx*(-.5*fe*ddnve*dnue-.5*fe*ddnue*dnve+penalty*dnue*dnve);
		    }
		}
	    }
	}
    }
    
/**
 * Auxiliary function computing the penalty parameter for interior
 * penalty methods on rectangles.
 *
 * Computation is done in two steps: first, we compute on each cell
 * <i>Z<sub>i</sub></i> the value <i>P<sub>i</sub> =
 * p<sub>i</sub>(p<sub>i</sub>+1)/h<sub>i</sub></i>, where <i>p<sub>i</sub></i> is
 * the polynomial degree on cell <i>Z<sub>i</sub></i> and
 * <i>h<sub>i</sub></i> is the length of <i>Z<sub>i</sub></i>
 * orthogonal to the current face.
 *
 * @author Guido Kanschat
 * @date 2010
 */
    template <int dim>
    double compute_penalty(
      const MeshWorker::DoFInfo<dim>& dinfo1,
      const MeshWorker::DoFInfo<dim>& dinfo2,
      unsigned int deg1,
      unsigned int deg2)
    {
      const unsigned int normal1 = GeometryInfo<dim>::unit_normal_direction[dinfo1.face_number];
      const unsigned int normal2 = GeometryInfo<dim>::unit_normal_direction[dinfo2.face_number];
      const unsigned int deg1sq = (deg1 == 0) ? 1 : deg1 * (deg1+1);
      const unsigned int deg2sq = (deg2 == 0) ? 1 : deg2 * (deg2+1);

      double penalty1 = deg1sq / dinfo1.cell->extent_in_direction(normal1);
      double penalty2 = deg2sq / dinfo2.cell->extent_in_direction(normal2);
      if (dinfo1.cell->has_children() ^ dinfo2.cell->has_children())
        {
          Assert (dinfo1.face == dinfo2.face, ExcInternalError());
          Assert (dinfo1.face->has_children(), ExcInternalError());
          penalty1 *= 8;
        }
      const double penalty = 0.5*(penalty1 + penalty2);
      return penalty;
    }
  }
}


DEAL_II_NAMESPACE_CLOSE

#endif
