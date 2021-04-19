/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2015 OpenFOAM Foundation
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
    computeInvariants

Group
    grpPostProcessingUtilities

Description
    Post-process Reynolds stresses to obtain invaraints.

    Compute:
    - normalized anisotropy
    - A2,A3 for Lumley triagnle
    - Coordinates in Barycentric triangle (xB, yB)

    Conventions following Eqs. (2.1)-(2.4) in (Emory & Iaccarino 2014)
    https://web.stanford.edu/group/ctr/ResBriefs/2014/14_emory.pdf
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"

#include "OSspecific.H"
#include "scalarList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
	"Compute invariants of Reynolds stress R: A2, A3, xB, yB"
    );

    // argList::noParallel();
    timeSelector::addOptions();

    #include "setRootCase.H"
    #include "createTime.H"

    // Get times list
    instantList timeDirs = timeSelector::select0(runTime, args);

    #include "createNamedMesh.H"

    IOdictionary invariantsDict
    (
        IOobject
        (
            "invariantsDict",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    );


    // For each time step read all fields
    forAll(timeDirs, timeI)
    {
        runTime.setTime(timeDirs[timeI], timeI);
        Info<< "Computing invariants for time " << runTime.timeName() << endl;

        #include "createFields.H"
        #include "calculateFields.H"

    }

    Info<< "\nEnd\n" << endl;

    return 0;
}


// ************************************************************************* //
