Info<< "Reading field U\n" << endl;

volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

// Velocity field is frozen. Compute gradU outside the loop
tmp<volTensorField> tgradU(fvc::grad(U));
const volTensorField& gradU = tgradU();


Info<< "Reading field R\n" << endl;

volSymmTensorField R
(
    IOobject
    (
        "R2",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

volSymmTensorField Pr
(
    IOobject
    (
        "Pr",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    -twoSymm(R & ((fvc::grad(U))()) )
);

dimensionedScalar kSmall("ks", dimVelocity*dimVelocity, SMALL);

volScalarField k
(
    IOobject
    (
        "k",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    0.5*tr(R)+kSmall
);
      


Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);


Info<< "Reading diffusivity and coefficients\n" << endl;

dimensionedScalar DRscalar("DR", dimViscosity, transportProperties);
dimensionedScalar delta("delta", dimLength, transportProperties);
dimensionedScalar nu("nu", dimViscosity, transportProperties);


dimensionedScalar C1
(
    dimensioned<scalar>::lookupOrAddToDict
    (
        "C1",
        transportProperties,
        1.8
    )
);



dimensionedScalar C2
(
    dimensioned<scalar>::lookupOrAddToDict
    (
        "C2",
        transportProperties,
        0.6
    )
);

dimensionedScalar Cmu
(
    dimensioned<scalar>::lookupOrAddToDict
    (
        "Cmu",
        transportProperties,
        0.09
    )
);

// epsilon = Cd k^(3/2) / lm
// nu_t = Cmu k^2/epsilon
// Cmu = c^4; Cd = c^3 --> Cc = Cmu^(1/4)
// see (Pope 2000) pp. 373; Eq. (10.44)-(10.47)
dimensionedScalar Cc(sqrt(sqrt(Cmu)));
dimensionedScalar Cd(pow(Cc, 3));

Switch initRwithChannel
(
    transportProperties.lookupOrAddDefault<Switch>("initRwithChannel", false)
);


Switch turbDiffusion
(
    transportProperties.lookupOrAddDefault<Switch>("turbDiffusion", true)
);


// mixing length: lm = min(κη, Cμδ)

// TODO: copy mixing length computation here:
// need to set up κ  and wall distance κη as well; Cmu is constructed above.

volScalarField d
(
    IOobject
    (
        "wallDistance",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    mesh
);

volScalarField lm
(
    IOobject
    (
        "mixingLength",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    min(0.41*d+100*SMALL*delta,0.09*delta)
);
lm.write();

// used only if using effective viscsity to diffuse R
volScalarField DREff
(
    IOobject
    (
        "DREff",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    DRscalar
);

// Don't write if DREff is specified by DR.
if(!turbDiffusion) {DREff.writeOpt() = IOobject::NO_WRITE;}

volScalarField epsilon
(
    IOobject
    (
        "epsilon",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    Cd*pow(sqrt(mag(k)), 3)/lm
);


if(initRwithChannel)
{
   // dimensionedScalar uprime // turb. intensity to initialize R
   //    (
   //         transportProperties.lookup("uprime")
   //     );
 
    dimensionedScalar uprime("uprime", dimVelocity, transportProperties);
    volScalarField nut(Cc * uprime * lm);
    volSymmTensorField twoS(twoSymm(gradU));
    volScalarField shear = nut*twoS.component(symmTensor::XY);

    // Follow Pope (2000), Fig. 7.17/7.33 for ratio of stress components
    // <u^2>:<v^2>:<w^2>:<uv> = 1:0.4:0.6:(-0.5)
    Info << "Replacing R with plane channel flow ratio ..." <<endl;
    R.replace(symmTensor::XY, shear);
    Info << "Replacing XX " <<endl;
    R.replace(symmTensor::XX, -2*shear);
    R.replace(symmTensor::YY, -0.8*shear);
    R.replace(symmTensor::ZZ, -1.2*shear);
    Info << "Replacing XZ " <<endl;
    R.replace(symmTensor::XZ, 0.0);
    Info << "Replacing YZ " <<endl;
    R.replace(symmTensor::YZ, 0.0);

    volSymmTensorField RChannel
        (
            "RChannel",
            R
        );
    RChannel.write();
}




#include "createPhi.H"

#include "createFvOptions.H"
