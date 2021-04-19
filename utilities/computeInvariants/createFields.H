    volSymmTensorField R
    (
        IOobject
        (
            "R2",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ
        ),
        mesh
    );
    
volScalarField A2
    (
        IOobject
        (
            "A2",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("A2v", dimless, 0.0)
    );

volScalarField A3
    (
        IOobject
        (
            "A3",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("A3v", dimless, 0.0)
    );


    volScalarField xB
    (
        IOobject
        (
            "xB",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("xbv", dimless, 0.0)
    );

    volScalarField yB
    (
        IOobject
        (
            "yB",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("ybv", dimless, 0.0)
    );


    volSymmTensorField b
    (
        IOobject
        (
            "b",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedSymmTensor("bv", dimless, symmTensor::zero)
    );

    volVectorField lams
    (
        IOobject
        (
            "lams",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedVector ("lamsv", dimless, vector::zero)
    );


