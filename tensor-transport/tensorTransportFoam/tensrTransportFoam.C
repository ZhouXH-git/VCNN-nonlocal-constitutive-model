/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
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
    tensorTransportFoam

Description
    Solves the steady or transient transport equation for a passive tensor (symmetric).

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating tensor transport\n" << endl;

    #include "CourantNo.H"

    // Velocity field is frozen. Compute gradU outside the loop
    tmp<volTensorField> tgradU(fvc::grad(U));
    const volTensorField& gradU = tgradU();

    
    if(initRwithChannel)
    {
        dimensionedScalar uprime // turb. intensity to initialize R
            (
                transportProperties.lookup("uprime")
            );

        volScalarField nut(Cc * uprime * lm);
        volSymmTensorField twoS(twoSymm(gradU));
        volScalarField shear = nut*twoS.component(symmTensor::XY);

        // Follow Pope (2000), Fig. 7.17/7.33 for ratio of stress components
        // <u^2>:<v^2>:<w^2>:<uv> = 1:0.4:0.6:(-0.5)
        Info << "Replacing R with plane channel flow ratio ..." <<endl;
        R.replace(tensor::XY, shear);
        R.replace(tensor::XX, -2*shear);
        R.replace(tensor::YY, -0.8*shear);
        R.replace(tensor::ZZ, -1.2*shear);
        R.replace(tensor::XZ, 0.0);
        R.replace(tensor::YZ, 0.0);

        volSymmTensorField RChannel
            (
                "RChannel",
                R
            );
        RChannel.write();
    }

    
    while (simple.loop(runTime))
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;


        
        while (simple.correctNonOrthogonal())
        {
            // Warning: R = <u_i, u_j> is -\tau !
            // This is LRR-IP model with
            // linear slow/rapid Pressure-Rate-of-Strain (PRoS) terms.

            Pr = -twoSymm(R & gradU);


            // Optional: Compute DR using eddy viscosity
            // TODO: uncomment the line below and test its effects
            // DREff = Cc * sqr(k) * lm + I*nu
            
            fvScalarMatrix REqn
            (
                fvm::ddt(R)
              + fvm::div(phi, R)
              - fvm::laplacian(DR, R)
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
            k = 0.5 * tr(R)
            epsilon = Cd * pow(sqrt(k), 3)/lm;
        }

        runTime.write();
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
