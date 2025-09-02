from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom  # For conformer generation
import sys

def generate_conformers(input_sdf, output_sdf, num_conformers=5000, optimize=True):
    """Generate an ensemble of conformers from a single SDF file.
    
    Args:
        input_sdf (str): Input SDF file (single molecule).
        output_sdf (str): Output SDF file (multiple conformers).
        num_conformers (int): Number of conformers to generate.
        optimize (bool): If True, optimize with MMFF94.
    """
    # Load the molecule
    mol = Chem.MolFromMolFile(input_sdf, removeHs=False)
    if mol is None:
        raise ValueError("Could not read molecule from SDF.")

    # Add hydrogens (if missing)
    mol = Chem.AddHs(mol)

    # Generate conformers
    conformer_ids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_conformers,
        pruneRmsThresh=0.5,  # Discard conformers with RMSD < 0.5 Å
        randomSeed=0xf00d,       # For reproducibility
    )

    # Optimize conformers (MMFF94)
    if optimize:
        for conf_id in conformer_ids:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)

    # Save to SDF
    writer = Chem.SDWriter(output_sdf)
    for conf_id in conformer_ids:
        writer.write(mol, confId=conf_id)
    writer.close()
    print(f"Generated {len(conformer_ids)} conformers in {output_sdf}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_conformers.py input.sdf output.sdf")
        sys.exit(1)
    generate_conformers(sys.argv[1], sys.argv[2])
