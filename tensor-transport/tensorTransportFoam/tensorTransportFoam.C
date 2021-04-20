/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    scalarTransportFoam

Group
    grpBasicSolvers

Description
    Passive scalar transport equation solver.

    \heading Solver details
    The equation is given by:

    \f[
        \ddt{T} + \div \left(\vec{U} T\right) - \div \left(D_T \grad T \right)
        = S_{T}
    \f]

    Where:
    \vartable
        T       | Passive scalar
        D_T     | Diffusion coefficient
        S_T     | Source
    \endvartable

    \heading Required fields
    \plaintable
        T       | Passive scalar
        U       | Velocity [m/s]
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Passive scalar transport equation solver."
    );

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating scalar transport\n" << endl;

    #include "CourantNo.H"


    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (simple.correctNonOrthogonal())
        {
             // Reynolds stress equation
             // Warning: R = <u_i, u_j> is -\tau !
            // This is LRR-IP model with
            // linear slow/rapid Pressure-Rate-of-Strain (PRoS) terms.

            Pr = -twoSymm(R & gradU);


            // Optional: Compute Effective DR using eddy viscosity
            if(turbDiffusion)
            {
                DREff = Cc * sqrt(k) * lm + I*nu;
            }

            fvSymmTensorMatrix REqn
            (
                fvm::ddt(R)
              + fvm::div(phi, R)
              - fvm::laplacian(DREff, R)
              + fvm::Sp(C1*epsilon/k, R) // anisotropic term of Rotta's model for slow PRoS
             ==
                Pr   // production
                - (2.0/3.0*(1 - C1)*I)*epsilon  // isotropic dissipation | isotropic term of Rotta's model for slow PRoS
                - C2*dev(Pr) // Rapid pressure-rate-of-strain (PRoS) term
                + fvOptions(R)
            );

            REqn.relax();
            fvOptions.constrain(REqn);
            REqn.solve();
            fvOptions.correct(R);
   
            //Bound normal stresses;
            scalar kMin(SMALL);
            R.max
                (
                    dimensionedSymmTensor
                    (
                        "zero",
                        R.dimensions(),
                        symmTensor
                        (
                            kMin, -GREAT, -GREAT,
                            kMin, -GREAT,

                            kMin
                        )
                    )
                );

            // update TKE and dissipation fields
            k = 0.5 * tr(R);
            epsilon = Cd * pow(sqrt(k), 3)/lm;

        }

        runTime.write();
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
