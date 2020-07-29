#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# (C) 2017 OpenEye Scientific Software Inc. All rights reserved.
#
# TERMS FOR USE OF SAMPLE CODE The software below ("Sample Code") is
# provided to current licensees or subscribers of OpenEye products or
# SaaS offerings (each a "Customer").
# Customer is hereby permitted to use, copy, and modify the Sample Code,
# subject to these terms. OpenEye claims no rights to Customer's
# modifications. Modification of Sample Code is at Customer's sole and
# exclusive risk. Sample Code may require Customer to have a then
# current license or subscription to the applicable OpenEye offering.
# THE SAMPLE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED.  OPENEYE DISCLAIMS ALL WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. In no event shall OpenEye be
# liable for any damages or liability in connection with the Sample Code
# or its use.

#############################################################################
# Visualizes the dihedral angles of a molecule
#############################################################################

import sys
import math
import glob
import uuid
from openeye import oechem
from openeye import oedepict
from openeye import oegrapheme
from perses.utils.openeye import createOEMolFromSDF
import os
import tqdm
import numpy as np

def main(argv=[__name__]):
    """
    itf = oechem.OEInterface()
    oechem.OEConfigure(itf, InterfaceData)
    if not oechem.OEParseCommandLine(itf, argv):
        return 1

    oname = itf.GetString("-out")
    iname = itf.GetString("-in")

    ext = oechem.OEGetFileExtension(oname)
    if not oedepict.OEIsRegisteredImageFile(ext):
        oechem.OEThrow.Fatal("Unknown image type!")

    ofs = oechem.oeofstream()
    if not ofs.open(oname):
        oechem.OEThrow.Fatal("Cannot open output file!")

  
    ## INPUT PARAMETERS
    #########################################################
    #########################################################
    
    mm = 'tyk2/og_pdbs'
    qml = 'tyk2/forward_snapshots'
    phase = 'solvent'
    which_ligand = 'old'
    dir_name = iname
    ligand_pdbs_mm = glob.glob(f"{mm}/{dir_name}/{which_ligand}*{phase}.pdb")
    print(len(ligand_pdbs_mm))
    ligand_pdbs_qml = glob.glob(f"{qml}/{dir_name}/{which_ligand}*{phase}.pdb")
    print(len(ligand_pdbs_qml))

    #d = np.load('full_data_dict.npy', allow_pickle=True)
    from_ligand, to_ligand = iname.replace('from', '').replace('to', '').replace('lig', '')
    print(from_ligand)
    print(to_ligand)
    #key1 = (1, 8)
    #key2 = ('solvent', which_ligand)
    #########################################################
    #########################################################

    #d = d.flatten()[0]
    #work = d[key1][key2]
    #print(work)

    
    for i, (mm_pdb_path, ani_pdb_path) in enumerate(zip(ligand_pdbs_mm, ligand_pdbs_qml)):
        print(mm_pdb_path, ani_pdb_path)
        if i == 0:
            MM_mol = createOEMolFromSDF(mm_pdb_path, 0)
            ANI_mol = createOEMolFromSDF(ani_pdb_path, 0)
        else:
            # there absolutely must be a better/faster way of doing this because this is ugly and slow
            MM_mol.NewConf(createOEMolFromSDF(mm_pdb_path, 0))
            ANI_mol.NewConf(createOEMolFromSDF(ani_pdb_path, 0))
""" 
    ofs = oechem.oeofstream()
    oname = f"tor_out"
    ext = oechem.OEGetFileExtension(oname)



    mm_pdb_path = f"og_lig0_solvent.pdb"
    ani_pdb_path = f"forward_lig0.solvent.pdb"
    MM_mol = createOEMolFromSDF(mm_pdb_path, 0)
    ANI_mol = createOEMolFromSDF(ani_pdb_path, 0)

   
    mol = MM_mol
    mol2 = ANI_mol

    for m in [mol, mol2]:
        oechem.OESuppressHydrogens(m)
        oechem.OECanonicalOrderAtoms(m)
        oechem.OECanonicalOrderBonds(m)
        m.Sweep()


    refmol = None

    stag = "dihedral_histogram"
    itag = oechem.OEGetTag(stag)

    nrbins = 20

    print(mol.NumConfs())
    print(mol2.NumConfs())

    get_dihedrals(mol, itag)
    set_dihedral_histograms(mol, itag, nrbins)

    get_dihedrals(mol2, itag)
    #set_weighted_dihedral_histograms(mol2, itag, work, nrbins)
    set_dihedral_histograms(mol2, itag, nrbins)

    width, height = 800, 400
    image = oedepict.OEImage(width, height)

    moffset = oedepict.OE2DPoint(0, 0)
    mframe = oedepict.OEImageFrame(image, width * 0.70, height, moffset)
    doffset = oedepict.OE2DPoint(mframe.GetWidth(), height * 0.30)
    dframe = oedepict.OEImageFrame(image, width * 0.30, height * 0.5, doffset)

    flexibility = True
    colorg = get_color_gradient(nrbins, flexibility)

    opts = oedepict.OE2DMolDisplayOptions(mframe.GetWidth(), mframe.GetHeight(),
                                          oedepict.OEScale_AutoScale)

    depict_dihedrals(mframe, dframe, mol, mol2, refmol, opts, itag, nrbins, colorg)

    if flexibility:
        lopts = oedepict.OELegendLayoutOptions(oedepict.OELegendLayoutStyle_HorizontalTopLeft,
                                               oedepict.OELegendColorStyle_LightBlue,
                                               oedepict.OELegendInteractiveStyle_Hover)
        lopts.SetButtonWidthScale(1.2)
        lopts.SetButtonHeightScale(1.2)
        lopts.SetMargin(oedepict.OEMargin_Right, 40.0)
        lopts.SetMargin(oedepict.OEMargin_Bottom, 80.0)

        legend = oedepict.OELegendLayout(image, "Legend", lopts)

        legend_area = legend.GetLegendArea()
        draw_color_gradient(legend_area, colorg)

        oedepict.OEDrawLegendLayout(legend)

    iconscale = 0.5
    oedepict.OEAddInteractiveIcon(image, oedepict.OEIconLocation_TopRight, iconscale)
    oedepict.OEDrawCurvedBorder(image, oedepict.OELightGreyPen, 10.0)

    oedepict.OEWriteImage(ofs, ext, image)

    return 0


class IsRotatableOrMacroCycleBond(oechem.OEUnaryBondPred):
    """
    Identifies rotatable bonds and single bonds in macro-cycles.
    """
    def __call__(self, bond):
        """
        :type mol: oechem.OEBondBase
        :rtype: boolean
        """
        if bond.GetOrder() != 1:
            return False
        if bond.IsAromatic():
            return False

        isrotor = oechem.OEIsRotor()
        if isrotor(bond):
            return True

        if oechem.OEBondGetSmallestRingSize(bond) >= 10:
            return True

        return False


def get_dihedrals(mol, itag):
    """
    Iterates over rotatable bonds and identifies their dihedral
    atoms. These atoms are added to the molecule in a group
    using the given tag.
    :type mol: oechem.OEMol
    :type itag: int
    :return: Number of dihedral angles identified
    :rtype: int
    """
    nrdihedrals = 0
    for bond in mol.GetBonds(IsRotatableOrMacroCycleBond()):
        atomB = bond.GetBgn()
        atomE = bond.GetEnd()

        neighB = None
        neighE = None

        for atom in atomB.GetAtoms(oechem.OEIsHeavy()):
            if atom != atomE:
                neighB = atom
                break
        for atom in atomE.GetAtoms(oechem.OEIsHeavy()):
            if atom != atomB:
                neighE = atom
                break

        if neighB is None or neighE is None:
            continue

        atomorder = [neighB, atomB, atomE, neighE]
        bondorder = [mol.GetBond(neighB, atomB), bond, mol.GetBond(neighE, atomE)]

        if neighB.GetIdx() < neighE.GetIdx():
            atomorder.reverse()
            bondorder.reverse()

        atoms = oechem.OEAtomVector(atomorder)
        bonds = oechem.OEBondVector(bondorder)

        nrdihedrals += 1
        mol.NewGroup(itag, atoms, bonds)

    return nrdihedrals


def set_dihedral_histograms(mol, itag, nrbins):
    """
    Iterates over the dihedral groups and bins the torsional
    angles for each conformation. The histogram data is then
    attached to the groups with the given tag.
    :type mol: oechem.OEMol
    :type itag: int
    :type nrbins: int
    """

    angleinc = 360.0 / float(nrbins)

    for group in mol.GetGroups(oechem.OEHasGroupType(itag)):
        atoms = oechem.OEAtomVector()
        for atom in group.GetAtoms():
            atoms.append(atom)
        histogram = [0] * nrbins

        for conf in mol.GetConfs():
            rad = oechem.OEGetTorsion(conf, atoms[0], atoms[1], atoms[2], atoms[3])
            deg = math.degrees(rad)
            deg = (deg + 360.0) % 360.0
            binidx = int(math.floor((deg / angleinc)))
            histogram[binidx] += 1

        histogram = list(normalize(np.array(histogram)))
        group.SetData(itag, histogram)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


def set_weighted_dihedral_histograms(mol, itag, work, nrbins):
    """
    Iterates over the dihedral groups and bins the torsional
    angles for each conformation. The histogram data is then
    attached to the groups with the given tag.
    :type mol: oechem.OEMol
    :type itag: int
    :type nrbins: int
    """


    angleinc = 360.0 / float(nrbins)

    # scale and normalize
    work = list(normalize(np.array(work) - min(work))) 

    for group in mol.GetGroups(oechem.OEHasGroupType(itag)):
        atoms = oechem.OEAtomVector()
        for atom in group.GetAtoms():
            atoms.append(atom)
        histogram = [0] * nrbins

        for idx, conf in enumerate(mol.GetConfs()):
            rad = oechem.OEGetTorsion(conf, atoms[0], atoms[1], atoms[2], atoms[3])
            deg = math.degrees(rad)
            deg = (deg + 360.0) % 360.0
            binidx = int(math.floor((deg / angleinc)))
            # instaed of 1 add the weight
            histogram[binidx] += np.exp(work[idx])

        group.SetData(itag, histogram)

def set_dihedral(mol, itag):
    """
    Iterates over the dihedral groups and attaches the
    dihedral angle to the group with the given tag.
    :type mol: oechem.OEMol
    :type itag: int
    """
    for group in mol.GetGroups(oechem.OEHasGroupType(itag)):
        atoms = oechem.OEAtomVector()
        for atom in group.GetAtoms():
            atoms.append(atom)

        rad = oechem.OEGetTorsion(mol, atoms[0], atoms[1], atoms[2], atoms[3])
        deg = math.degrees(rad)
        deg = (deg + 360.0) % 360.0
        group.SetData(itag, deg)


def depict_dihedrals(image, dimage, mol1, mol2, refmol, opts, itag, nrbins, colorg):
    """
    Highlights the dihedral atoms of a torsion and the depicts the
    corresponding dihedral angle histogram when hovering over
    the center of the torsion on the molecule display.
    :type image: oedepict.OEImageBase
    :type dimage: oedepict.OEImageBase
    :type mol: oechem.OEMol
    :type refmol: oechem.OEMol
    :type opts: oedepict.OE2DMolDisplayOptions
    :type itag: int
    :type nrbins: int
    :type oechem.OEColorGradientBase
    """

    nrconfs = mol1.NumConfs()
    center = oedepict.OEGetCenter(dimage)
    radius = min(dimage.GetWidth(), dimage.GetHeight()) * 0.40

    draw_dihedral_circle(dimage, center, radius, nrbins, nrconfs)

    suppressH = True
    oegrapheme.OEPrepareDepictionFrom3D(mol1, suppressH)
    if refmol is not None:
        oegrapheme.OEPrepareDepictionFrom3D(refmol, suppressH)

    disp = oedepict.OE2DMolDisplay(mol1, opts)

    dihedrals = []
    ref_dihedrals = []
    centers = []
    agroups = []
    dgroups = []

    dihedrals_ref_dist = []
    for group in mol2.GetGroups(oechem.OEHasGroupType(itag)):

        dihedrals_ref_dist.append(group)

    nrdihedrals = 0
    for group in mol1.GetGroups(oechem.OEHasGroupType(itag)):

        uniqueid = uuid.uuid4().hex
        agroup = image.NewSVGGroup("torsion_area_" + uniqueid)
        dgroup = image.NewSVGGroup("torsion_data_" + uniqueid)
        oedepict.OEAddSVGHover(agroup, dgroup)

        dihedrals.append(group)
        if refmol is not None:
            ref_dihedrals.append(get_reference_dihedral(group, refmol, itag))

        centers.append(get_center(disp, group))
        agroups.append(agroup)
        dgroups.append(dgroup)
        nrdihedrals += 1

    for didx in range(0, nrdihedrals):

        image.PushGroup(dgroups[didx])

        dihedral = dihedrals[didx]
        abset = oechem.OEAtomBondSet(dihedral.GetAtoms(), dihedral.GetBonds())
        draw_highlight(image, disp, abset)
        dihedral_histogram = dihedral.GetData(itag)
        dihedral_histogram_ref = dihedrals_ref_dist[didx].GetData(itag)

        print(dihedral_histogram)
        print(dihedral_histogram_ref)
        draw_dihedral_histogram(dimage, dihedral_histogram, dihedral_histogram_ref, center, radius, nrbins, nrconfs)

        image.PopGroup(dgroups[didx])

    clearbackground = True
    oedepict.OERenderMolecule(image, disp, not clearbackground)

    markpen = oedepict.OEPen(oechem.OEBlack, oechem.OEWhite, oedepict.OEFill_On, 1.0)
    farpen = oedepict.OEPen(oechem.OEBlack, oechem.OERed, oedepict.OEFill_Off, 2.0)

    angleinc = 360.0 / float(nrbins)

    for didx in range(0, nrdihedrals):

        image.PushGroup(agroups[didx])

        dihedral = dihedrals[didx]
        dihedral_histogram = dihedral.GetData(itag)
        flexibility = determine_flexibility(dihedral_histogram)
        color = colorg.GetColorAt(flexibility)
        markpen.SetBackColor(color)

        markradius = disp.GetScale() / 8.0
        image.DrawCircle(centers[didx], markradius, markpen)

        if refmol is not None and ref_dihedrals[didx] is not None:
            ref_dihedral = ref_dihedrals[didx]
            if get_closest_dihedral_angle(mol1, dihedral, ref_dihedral, itag) > angleinc:
                image.DrawCircle(centers[didx], markradius, farpen)

        radius = disp.GetScale() / 4.0
        image.DrawCircle(centers[didx], radius, oedepict.OESVGAreaPen)

        image.PopGroup(agroups[didx])


def get_closest_dihedral_angle(mol, dihedral, ref_dihedral, itag):
    """
    Returns the closest torsion angle difference to the reference.
    :type mol: oechem.OEMol
    :type dihedral: oechem.OEGroupBase
    :type ref_dihedral: oechem.OEGroupBase
    :type itag: int
    """

    closest_angle = float("inf")

    for conf in mol.GetConfs():
        atoms = [a for a in dihedral.GetAtoms()]
        rad = oechem.OEGetTorsion(conf, atoms[0], atoms[1], atoms[2], atoms[3])
        deg = math.degrees(rad)
        angle_diff = (abs(deg - ref_dihedral.GetData(itag)) + 360) % 360
        closest_angle = min(closest_angle, angle_diff)

    return closest_angle


def get_center(disp, dgroup):
    """
    Returns the center of a dihedral angle (stored in a group) on the
    molecule display.
    :type disp: oedepict.OE2DMolDisplay
    :type dgroup: oechem.OEGroupBase
    """

    for bond in dgroup.GetBonds():
        atomB = bond.GetBgn()
        atomE = bond.GetEnd()

        nrneighsB = 0
        for neigh in atomB.GetAtoms():
            if dgroup.HasAtom(neigh):
                nrneighsB += 1
        nrneighsE = 0
        for neigh in atomE.GetAtoms():
            if dgroup.HasAtom(neigh):
                nrneighsE += 1

        if nrneighsB != 2 or nrneighsE != 2:
            continue

        adispB = disp.GetAtomDisplay(atomB)
        adispE = disp.GetAtomDisplay(atomE)
        return (adispB.GetCoords() + adispE.GetCoords()) / 2.0


def draw_dihedral_circle(image, center, radius, nrbins, nrconfs):
    """
    Draws the base radial histogram.
    :type image: oedepict.OEImageBase
    :type center: oedepict.OE2DPoint
    :type radius: float
    :type nrbins: int
    :type nrconfs: int
    """

    grey = oechem.OEColor(210, 210, 210)
    pen = oedepict.OEPen(grey, grey, oedepict.OEFill_On, 1.0)
    image.DrawCircle(center, radius, pen)

    linegrey = oechem.OEColor(220, 220, 220)
    linepen = oedepict.OEPen(linegrey, linegrey, oedepict.OEFill_On, 1.0)

    angleinc = 360.0 / float(nrbins)

    v = oedepict.OE2DPoint(0.0, -1.0)
    for i in range(0, nrbins):
        end = oedepict.OELengthenVector(oedepict.OERotateVector(v, i * angleinc), radius)
        image.DrawLine(center, center + end, linepen)

    fontsize = int(math.floor(radius * 0.1))
    font = oedepict.OEFont(oedepict.OEFontFamily_Default, oedepict.OEFontStyle_Bold,
                           fontsize, oedepict.OEAlignment_Center, oechem.OEBlack)

    for i in range(0, 4):
        angle = i * 90.0
        end = oedepict.OELengthenVector(oedepict.OERotateVector(v, angle), radius * 1.20)
        text = '{:.1f}'.format(angle)
        dim = radius / 2.5
        textframe = oedepict.OEImageFrame(image, dim, dim,
                                          center + end - oedepict.OE2DPoint(dim / 2.0, dim / 2.0))
        oedepict.OEDrawTextToCenter(textframe, text, font)

    minradius = radius / 3.0
    whitepen = oedepict.OEPen(oechem.OEWhite, oechem.OEWhite, oedepict.OEFill_On, 1.0, oedepict.OEStipple_NoLine)
    image.DrawCircle(center, minradius, whitepen)

    font.SetSize(int(fontsize * 1.5))
    top = oedepict.OE2DPoint(image.GetWidth() / 2.0, - 10.0)
    image.DrawText(top, 'torsion histogram', font)
    top = oedepict.OE2DPoint(image.GetWidth() / 2.0, - 30.0)

    image.DrawText(top, 'MM: blue; ANI: red', font)


    bottom = oedepict.OE2DPoint(image.GetWidth() / 2.0, image.GetHeight() + 26.0)
    image.DrawText(bottom, 'number of conformations: {}'.format(nrconfs), font)


def get_text_angle(angle):
    if angle <= 180.0:
        return (360 - angle + 90.0) % 360
    else:
        return (180 - angle + 90.0) % 360


def draw_dihedral_histogram(image, histogram, histogram_ref, center, radius, nrbins, nrconfs):
    """
    Draws the radial histogram of a torsional angle.
    :type image: oedepict.OEImageBase
    :type histogram: list(int)
    :type center: oedepict.OE2DPoint
    :type radius: float
    :type nrbins: int
    :type nrconfs: int
    :type nrbins: int
    """

    minradius = radius / 3.0
    maxvalue = max(max(histogram), max(histogram_ref))
    radiusinc = (radius - minradius) / maxvalue

    angleinc = 360.0 / float(nrbins)

    valuepen = oedepict.OEPen(oechem.OERoyalBlue, oechem.OERoyalBlue, oedepict.OEFill_On, 2.0)

    maxvalue = 0
    maxvalueidx = 0
    for i in range(0, len(histogram)):
        value = histogram[i]
        if value == 0:
            continue

        if value > maxvalue:
            maxvalue = value
            maxvalueidx = i

        arcradius = value * radiusinc + minradius
        if arcradius < 1.0:
            continue

        bgnangle = i * angleinc
        endangle = (i + 1) * angleinc

        image.DrawPie(center, bgnangle, endangle, arcradius, valuepen)

    valuepen = oedepict.OEPen(oechem.OERed, oechem.OERed, oedepict.OEFill_Off, 2.0)
    for i in range(0, len(histogram_ref)):
        value = histogram_ref[i]
        if value == 0:
            continue

        if value > maxvalue:
            maxvalue = value
            maxvalueidx = i

        arcradius = value * radiusinc + minradius
        if arcradius < 1.0:
            continue

        bgnangle = i * angleinc
        endangle = (i + 1) * angleinc

        image.DrawPie(center, bgnangle, endangle, arcradius, valuepen)

    percent = maxvalue / (nrconfs / 100.0)

    whitepen = oedepict.OEPen(oechem.OEWhite, oechem.OEWhite, oedepict.OEFill_On, 0.2, oedepict.OEStipple_NoLine)
    image.DrawCircle(center, minradius, whitepen)

    fontsize = int(math.floor(radius * 0.1))
    font = oedepict.OEFont(oedepict.OEFontFamily_Default, oedepict.OEFontStyle_Bold,
                           fontsize, oedepict.OEAlignment_Center, oechem.OEWhite)
    angle = maxvalueidx * angleinc
    if angle >= 180.0:
        angle += angleinc * 0.3
    else:
        angle += angleinc * 0.7
    textangle = get_text_angle(angle)
    v = oedepict.OE2DPoint(0.0, -1.0)
    pos = oedepict.OELengthenVector(oedepict.OERotateVector(v, angle), radius * 0.80)
    font.SetRotationAngle(textangle)
    image.DrawText(center + pos, "{:.1f}%".format(percent*100), font)


def are_same_groups(agroup, bgroup):

    for a, b in zip(agroup.GetAtoms(), bgroup.GetAtoms()):
        if a.GetIdx() != b.GetIdx():
            return False

    for a, b in zip(agroup.GetBonds(), bgroup.GetBonds()):
        if a.GetIdx() != b.GetIdx():
            return False

    return True


def get_reference_dihedral(group, refmol, itag):
    """
    Returns the torsion angle on the reference molecule that
    corresponds to the torsional angle of the multi-conformer
    molecule.
    :type group: oechem.OEGroupBase
    :type refmol: oechem.OEMol
    :type itag: int
    """
    if refmol is None:
        return None

    for refgroup in refmol.GetGroups(oechem.OEHasGroupType(itag)):
        if are_same_groups(group, refgroup):
            return refgroup

    return None


def draw_reference_dihedral(image, group, itag, center, radius):
    """
    Draws dihedral angle of the reference molecule.
    :type image: oedepict.OEImageBase
    :type group: oechem.OEGroupBase
    :type itag: int
    :type center: oedepict.OE2DPoint
    :type radius: float
    """

    if not group.HasData(itag):
        return
    angle = group.GetData(itag)
    v = oedepict.OE2DPoint(0.0, -1.0)
    bgn = oedepict.OELengthenVector(oedepict.OERotateVector(v, angle), radius / 6.0)
    end = oedepict.OELengthenVector(oedepict.OERotateVector(v, angle), radius / 3.0)
    redpen = oedepict.OEPen(oechem.OERed, oechem.OERed, oedepict.OEFill_Off, 2.0)
    image.DrawLine(center + bgn, center + end, redpen)

    fontsize = int(math.floor(radius * 0.12))
    font = oedepict.OEFont(oedepict.OEFontFamily_Default, oedepict.OEFontStyle_Bold,
                           fontsize, oedepict.OEAlignment_Center, oechem.OERed)

    dim = radius / 2.5
    textframe = oedepict.OEImageFrame(image, dim, dim,
                                      center - oedepict.OE2DPoint(dim / 2.0, dim / 2.0))
    oedepict.OEDrawTextToCenter(textframe, "{:.1f}".format(angle), font)


def draw_highlight(image, disp, abset):
    """
    Highlights the atoms of the dihedral angle on the molecule display.
    :type image: oedepict.OEImageBase
    :type disp: oedepict.OE2DMolDisplay
    :type abset: oechem.OEAtomBondSet
    """

    linewidth = disp.GetScale() / 2.0
    pen = oedepict.OEPen(oechem.OEBlueTint, oechem.OEBlueTint, oedepict.OEFill_On, linewidth)
    for bond in abset.GetBonds():
        adispB = disp.GetAtomDisplay(bond.GetBgn())
        adispE = disp.GetAtomDisplay(bond.GetEnd())
        image.DrawLine(adispB.GetCoords(), adispE.GetCoords(), pen)


def get_color_gradient(nrbins, flexibility):
    """
    Initializes the color gradient used to color the circle in the middle of
    the rotatable bond.
    :type nrbins: int
    :type flexibility: boolean
    """

    colorg = oechem.OEExponentColorGradient(0.25)

    if flexibility:
        colorg.AddStop(oechem.OEColorStop(1, oechem.OEBlack))
        colorg.AddStop(oechem.OEColorStop(nrbins, oechem.OERed))
    else:
        colorg.AddStop(oechem.OEColorStop(1, oechem.OEBlack))
        colorg.AddStop(oechem.OEColorStop(nrbins, oechem.OEBlack))

    return colorg


def draw_color_gradient(image, colorg):
    """
    Draws the color gradient used to color the circle in the middle of
    the rotatable bond.
    :type image: oedepict.OEImageBase
    :type colorg: oechem.OEColorGradientBase
    """

    width, height = image.GetWidth(), image.GetHeight()
    frame = oedepict.OEImageFrame(image, width * 0.8, height * 0.8,
                                  oedepict.OE2DPoint(width * 0.1, height * 0.1))

    opts = oegrapheme.OEColorGradientDisplayOptions()
    opts.SetColorStopPrecision(1)
    opts.SetColorStopLabelFontScale(0.5)
    opts.SetColorStopVisibility(False)

    opts.AddLabel(oegrapheme.OEColorGradientLabel(colorg.GetMinValue(), "rigid"))
    opts.AddLabel(oegrapheme.OEColorGradientLabel(colorg.GetMaxValue(), "flexible"))

    oegrapheme.OEDrawColorGradient(frame, colorg, opts)


def determine_flexibility(histogram):
    """
    Returns the simple estimation of torsion flexibility by counting the
    number of non-zero bins in the torsional histogram.
    :type histogram: list(int)
    """

    nr_non_zero_bins = sum([1 for x in histogram if x > 0]) * 2
    return nr_non_zero_bins


InterfaceData = '''
!CATEGORY "input/output options"
    !PARAMETER -in
      !ALIAS -o
      !TYPE string
      !REQUIRED true
      !KEYLESS 1
      !VISIBILITY simple
      !BRIEF Input filename for the system
    !END
    !PARAMETER -out
      !ALIAS -o
      !TYPE string
      !REQUIRED true
      !KEYLESS 2
      !VISIBILITY simple
      !BRIEF Output filename of the generated image
    !END
!END
!CATEGORY "visualization options"
    !PARAMETER -nrbins
      !TYPE int
      !REQUIRED true
      !DEFAULT 24
      !LEGAL_RANGE 12 36
      !VISIBILITY simple
      !BRIEF Number of bins in the dihedral angle histogram
    !END
    !PARAMETER -flexibility
      !ALIAS -f
      !TYPE  bool
      !REQUIRED true
      !DEFAULT false
      !VISIBILITY simple
      !BRIEF Visualize dihedral angle flexibility
    !END
!END
'''


if __name__ == "__main__":
    sys.exit(main(sys.argv))

