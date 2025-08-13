import sys
import os
import pkg_resources

print("Python executable:", sys.executable)
print("System PATH:", os.environ.get("PATH", ""))
installed_packages = {pkg.key for pkg in pkg_resources.working_set}
print("Installed packages:", installed_packages)

import gradio as gr
import py3Dmol
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import openmm as mm
from openmm.app import *
from openmm import *
from openmm.unit import *
import matplotlib.pyplot as plt
from biopandas.pdb import PandasPdb
import nglview as nv
from IPython.display import display
import mdtraj as md
from boltons.cacheutils import LRU
import os
import zipfile

# --- Configuration ---
CACHE = LRU(max_size=128)

# --- Helper Functions ---
def fetch_pdb(pdb_id):
    """Fetches a PDB file from the RCSB PDB database."""
    if len(pdb_id) != 4:
        raise gr.Error("PDB ID must be 4 characters long.")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        pdb_content = response.text
        CACHE[f"pdb_{pdb_id}"] = pdb_content
        return pdb_content
    else:
        raise gr.Error(f"Failed to fetch PDB ID: {pdb_id}")

def read_pdb_file(pdb_file):
    """Reads a PDB file from a file upload."""
    with open(pdb_file.name, "r") as f:
        pdb_content = f.read()
    CACHE["pdb_upload"] = pdb_content
    return pdb_content

def visualize_protein(pdb_content, highlight_pockets=None, highlight_ligand=None):
    """Visualizes a protein structure using py3Dmol."""
    if not pdb_content:
        return None

    view = py3Dmol.view(width=600, height=400)
    view.addModel(pdb_content, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})

    if highlight_pockets:
        for i, pocket in enumerate(highlight_pockets):
            color = f"0x{i*20:02x}{i*20:02x}{255-i*20:02x}"
            view.addSphere({
                "center": {"x": pocket['x'], "y": pocket['y'], "z": pocket['z']},
                "radius": pocket['radius'],
                "color": color,
                "alpha": 0.7
            })
            view.addLabel(f"Pocket {i+1}", {"position": {"x": pocket['x'], "y": pocket['y'], "z": pocket['z']}})


    if highlight_ligand:
        view.addModel(highlight_ligand, "sdf")
        view.setStyle({"model": 1}, {"stick": {}})


    view.zoomTo()
    return view.render()

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Automated Drug Discovery Pipeline")

    with gr.Tabs():
        with gr.TabItem("1. PDB Input & Preview"):
            with gr.Row():
                pdb_id_input = gr.Textbox(label="PDB ID (e.g., 6M0J)")
                pdb_upload = gr.File(label="Or Upload PDB File")
                download_pdb_button = gr.Button("Download PDB")

            protein_viewer = gr.HTML()
            pdb_download_file = gr.File(label="Download PDB File")

            def process_pdb_input(pdb_id, pdb_upload):
                if pdb_id:
                    pdb_content = fetch_pdb(pdb_id)
                elif pdb_upload:
                    pdb_content = read_pdb_file(pdb_upload)
                else:
                    return None, None
                return visualize_protein(pdb_content), None

            def download_pdb(pdb_id):
                if pdb_id:
                    pdb_content = fetch_pdb(pdb_id)
                    with open(f"{pdb_id}.pdb", "w") as f:
                        f.write(pdb_content)
                    return f"{pdb_id}.pdb"
                else:
                    return None

            pdb_id_input.change(process_pdb_input, inputs=[pdb_id_input, pdb_upload], outputs=[protein_viewer, pdb_download_file])
            pdb_upload.change(process_pdb_input, inputs=[pdb_id_input, pdb_upload], outputs=[protein_viewer, pdb_download_file])
            download_pdb_button.click(download_pdb, inputs=pdb_id_input, outputs=pdb_download_file)

        with gr.TabItem("2. Protein Preparation"):
            with gr.Row():
                run_pdbfixer_button = gr.Button("Run PDBFixer")
            
            protein_comparison_viewer = gr.HTML()

            def run_protein_preparation(pdb_id, pdb_upload):
                pdb_content = CACHE.get(f"pdb_{pdb_id}") or CACHE.get("pdb_upload")
                if not pdb_content:
                    raise gr.Error("Please provide a PDB file in the previous step.")

                # Save the original PDB content to a temporary file
                with open("original.pdb", "w") as f:
                    f.write(pdb_content)

                fixer = pdbfixer(filename="original.pdb")
                fixer.findMissingResidues()
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                fixer.addMissingHydrogens(7.0)

                # Save the fixed PDB content to a new file
                with open("fixed.pdb", "w") as f:
                    PDBFile.writeFile(fixer.topology, fixer.positions, f)

                with open("fixed.pdb", "r") as f:
                    fixed_pdb_content = f.read()

                CACHE["fixed_pdb"] = fixed_pdb_content

                # Visualize the before and after
                view = py3Dmol.view(width=600, height=400)
                view.addModel(pdb_content, "pdb")
                view.setStyle({"model": 0}, {"cartoon": {"color": "blue"}})
                view.addModel(fixed_pdb_content, "pdb")
                view.setStyle({"model": 1}, {"cartoon": {"color": "red"}})
                view.zoomTo()
                return view.render()

            run_pdbfixer_button.click(run_protein_preparation, inputs=[pdb_id_input, pdb_upload], outputs=protein_comparison_viewer)

        with gr.TabItem("3. Pocket Detection"):
            with gr.Row():
                run_fpocket_button = gr.Button("Run fpocket")
            
            pocket_viewer = gr.HTML()

            def run_fpocket_mock(pdb_id, pdb_upload):
                fixed_pdb_content = CACHE.get("fixed_pdb")
                if not fixed_pdb_content:
                    raise gr.Error("Please run protein preparation in the previous step.")

                # Mock fpocket results
                pockets = [
                    {'x': 10, 'y': 10, 'z': 10, 'radius': 5, 'score': 0.9},
                    {'x': 20, 'y': 20, 'z': 20, 'radius': 6, 'score': 0.8},
                    {'x': 30, 'y': 30, 'z': 30, 'radius': 7, 'score': 0.7},
                    {'x': 40, 'y': 40, 'z': 40, 'radius': 8, 'score': 0.6},
                    {'x': 50, 'y': 50, 'z': 50, 'radius': 9, 'score': 0.5},
                ]

                CACHE["pockets"] = pockets

                return visualize_protein(fixed_pdb_content, highlight_pockets=pockets)

            run_fpocket_button.click(run_fpocket_mock, inputs=[pdb_id_input, pdb_upload], outputs=pocket_viewer)

        with gr.TabItem("4. Ligand Preparation"):
            with gr.Row():
                ligand_csv_upload = gr.File(label="Upload CSV with SMILES")
                generate_ligands_button = gr.Button("Generate Mock Ligands")
            
            ligand_gallery = gr.Gallery(label="2D Ligand Structures")
            ligand_viewer_3d = gr.HTML()

            def process_ligands(ligand_csv, generate_mock):
                if ligand_csv is not None:
                    df = pd.read_csv(ligand_csv.name)
                    smiles_list = df["SMILES"].tolist()
                elif generate_mock:
                    smiles_list = ["CCO", "CCN", "CNC"]
                else:
                    return None, None

                CACHE["smiles_list"] = smiles_list

                images = []
                molecules_3d = []
                for smiles in smiles_list:
                    mol = Chem.MolFromSmiles(smiles)
                    images.append(Draw.MolToImage(mol))
                    
                    mol_3d = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
                    molecules_3d.append(Chem.MolToMolBlock(mol_3d))

                CACHE["molecules_3d"] = molecules_3d

                # Visualize the 3D ligands
                view = py3Dmol.view(width=600, height=400)
                for mol_3d_sdf in molecules_3d:
                    view.addModel(mol_3d_sdf, "sdf")
                view.setStyle({}, {"stick": {}})
                view.zoomTo()
                
                return images, view.render()

            ligand_csv_upload.change(process_ligands, inputs=[ligand_csv_upload, generate_ligands_button], outputs=[ligand_gallery, ligand_viewer_3d])
            generate_ligands_button.click(process_ligands, inputs=[ligand_csv_upload, generate_ligands_button], outputs=[ligand_gallery, ligand_viewer_3d])

        with gr.TabItem("5. Docking"):
            with gr.Row():
                run_docking_button = gr.Button("Run Docking")
            
            docking_results_table = gr.DataFrame(headers=["Ligand", "Pocket", "Score"])
            docking_viewer = gr.HTML()

            def run_docking():
                fixed_pdb_content = CACHE.get("fixed_pdb")
                pockets = CACHE.get("pockets")
                molecules_3d = CACHE.get("molecules_3d")
                smiles_list = CACHE.get("smiles_list")

                if not all([fixed_pdb_content, pockets, molecules_3d, smiles_list]):
                    raise gr.Error("Please complete the previous steps.")

                results = []
                for i, pocket in enumerate(pockets):
                    for j, mol_3d_sdf in enumerate(molecules_3d):
                        # Mock docking score
                        score = -1 * (i + j + 1) * 0.1
                        results.append([smiles_list[j], f"Pocket {i+1}", score])

                df = pd.DataFrame(results, columns=["Ligand", "Pocket", "Score"])
                df = df.sort_values(by="Score").reset_index(drop=True)

                CACHE["docking_results"] = df

                # Visualize the best pose
                best_ligand_smiles = df.iloc[0]["Ligand"]
                best_ligand_index = smiles_list.index(best_ligand_smiles)
                best_ligand_sdf = molecules_3d[best_ligand_index]

                return df, visualize_protein(fixed_pdb_content, highlight_pockets=pockets, highlight_ligand=best_ligand_sdf)

            run_docking_button.click(run_docking, outputs=[docking_results_table, docking_viewer])

        with gr.TabItem("6. Molecular Dynamics"):
            with gr.Row():
                run_md_button = gr.Button("Run MD Simulation")
            
            md_plots = gr.Plot()
            md_viewer = gr.HTML()

            def run_md_simulation():
                docking_results = CACHE.get("docking_results")
                if docking_results is None:
                    raise gr.Error("Please run docking first.")

                # Mock MD simulation
                time = [i for i in range(100)]
                rmsd = [0.1 + i*0.01 for i in range(100)]
                rmsf = [0.2 + i*0.005 for i in range(100)]
                energy = [-1000 + i*5 for i in range(100)]

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
                ax1.plot(time, rmsd)
                ax1.set_title("RMSD")
                ax2.plot(time, rmsf)
                ax2.set_title("RMSF")
                ax3.plot(time, energy)
                ax3.set_title("Energy")
                plt.tight_layout()

                # Mock trajectory visualization
                fixed_pdb_content = CACHE.get("fixed_pdb")
                best_ligand_smiles = docking_results.iloc[0]["Ligand"]
                best_ligand_index = CACHE.get("smiles_list").index(best_ligand_smiles)
                best_ligand_sdf = CACHE.get("molecules_3d")[best_ligand_index]

                view = py3Dmol.view(width=600, height=400)
                view.addModel(fixed_pdb_content, "pdb")
                view.setStyle({"cartoon": {"color": "spectrum"}})
                view.addModel(best_ligand_sdf, "sdf")
                view.setStyle({"model": 1}, {"stick": {}})
                view.zoomTo()

                return fig, view.render()

            run_md_button.click(run_md_simulation, outputs=[md_plots, md_viewer])

        with gr.TabItem("7. Final Results"):
            with gr.Row():
                generate_results_button = gr.Button("Generate Final Results")
                download_button = gr.Button("Download Results")
            
            final_results_table = gr.DataFrame()
            final_viewer = gr.HTML()

            def generate_final_results():
                docking_results = CACHE.get("docking_results")
                if docking_results is None:
                    raise gr.Error("Please run docking first.")

                # Mock ADME/T and MD stability metrics
                admet_results = pd.DataFrame({
                    "Ligand": docking_results["Ligand"],
                    "ADME_Score": [0.8, 0.7, 0.9, 0.6, 0.5] * (len(docking_results) // 5 + 1),
                    "Toxicity": ["Low", "High", "Low", "Medium", "Low"] * (len(docking_results) // 5 + 1)
                })
                md_stability = pd.DataFrame({
                    "Ligand": docking_results["Ligand"],
                    "RMSD_avg": [0.15, 0.2, 0.1, 0.25, 0.3] * (len(docking_results) // 5 + 1),
                    "RMSF_avg": [0.25, 0.3, 0.2, 0.35, 0.4] * (len(docking_results) // 5 + 1)
                })

                final_df = pd.merge(docking_results, admet_results, on="Ligand")
                final_df = pd.merge(final_df, md_stability, on="Ligand")
                final_df = final_df.drop_duplicates(subset=["Ligand","Pocket"])

                CACHE["final_results"] = final_df

                return final_df, visualize_protein(CACHE.get("fixed_pdb"))

            def download_results():
                with zipfile.ZipFile("drug_discovery_results.zip", "w") as zf:
                    if CACHE.get("fixed_pdb"):
                        zf.writestr("fixed_protein.pdb", CACHE.get("fixed_pdb"))
                    if CACHE.get("docking_results") is not None:
                        zf.writestr("docking_results.csv", CACHE.get("docking_results").to_csv())
                    if CACHE.get("final_results") is not None:
                        zf.writestr("final_results.csv", CACHE.get("final_results").to_csv())
                return "drug_discovery_results.zip"

            generate_results_button.click(generate_final_results, outputs=[final_results_table, final_viewer])
            download_button.click(download_results, outputs=gr.File(label="Download Results"))


if __name__ == "__main__":
    demo.launch()
