# PPAP
The model integrates structural features with sequence representations through an interfacial contact-aware attention mechanism. specially, it extracts sequence features from ESM and processes interaction interface information through a decoder. Features from protein complex interactions, including edge and node attributes, are concatenated and passed through a linear layer, followed by an interfacial contact-aware attention mechanism that determines interaction type and strength. The graph representation is then processed through a feedforward network (FFN) to predict the affinity.   


![image](https://github.com/TEKHOO/PPAP/blob/main/PPAP.png)
## Dataset
The figure illustrates the selection process for protein-protein complexes, categorized into dimers, oligomeric complexes, and complexes containing three or more proteins. Different strategies, including pair wise chain selection, EPPIC prediction, and manual selection, are applied based on complex type. The right side shows structural examples, with final selected chains highlighted in blue.  

![image2](https://github.com/TEKHOO/PPAP/blob/main/data_process.png)

The latest protein-protein interaction chain information can be found in  
`PPAP/dataset/PDBbind_v2021_chain_info.txt`

## Software Prerequisites
* [ESM2-3B](https://github.com/facebookresearch/esm) - We use esm2_t36_3B_UR50D to get protein embeddings.
* [H5py](https://docs.h5py.org/en/stable/quick.html#quick) ≥ 3.11.0
* [Pytorch](https://pytorch.org/) ≥ 1.12.0
* [Pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) ≥ 2.1
* [Torch-geometric](https://github.com/pyg-team/pytorch_geometric) ≥ 2.5.3
* [Torchmetrics](https://lightning.ai/docs/torchmetrics/stable/) ≥ 0.9.3

    To replicate the Conda environment used in this project, follow the steps below:

    Run the following code:
    ```bash
    
    conda env create -f environment.yml
    conda activate PPAP

    ```
    Install PyTorch that matches your CUDA version.
    ```bash
    # CUDA 10.2
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
    # CUDA 11.3
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
    # CUDA 11.6
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
    ```
    If your CUDA version is ≥ 12.0, you also need to run:
    ```bash
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
    pip install torch==2.3.1

    ```
    If none of the above installation methods work, you can download our pre-configured environment ​​PPAP.tar.gz​​ from [huggingface](https://huggingface.co/qj666/PPAP/tree/main), then move it to your Conda environments directory.
    

## Make predictions
1. **Download Model Weights**  

    Model weights can be downloaded from [huggingface](https://huggingface.co/qj666/PPAP/tree/main).  

    Place them in the `PPAP/weight/`.
    
    ```bash
    cp your_weight_path/epoch=11-val_r2=0.361.ckpt PPAP/weight/
    ```

2. **Input PDB Files**
   
    Place the PDB files in `/PPAP/input/pdb`.

    ⚠️ It is recommended that PDB filenames (protein names) do not contain the "_" character to avoid unexpected errors.

3. **Input Receptor-Ligand Information**  

    Edit the `PPAP/input/chain_info.txt` file to provide the interaction chain information of the proteins to be tested.  
   
    Each line should include the protein name and its interaction chains, such as:

    ```text
    1a4y_A_B
    3a67_HL_Y
    3d3v_ABC_DE
    ...
    ```

4. **Calculate Node Features and Edge Features**  

    Run `cal_graph.py`:
    
    ```bash
    cd PPAP/strips/
    python cal_graph.py
    ```

5. **Run the Model**  

    Run `PPAP_test.py`:
    
    ```bash
    cd PPAP
    python PPAP_test.py
    ```

6. **View the Output**

    Check the output results in `PPAP/output/result.xlsx`, which provides the two outputs: -ΔG (kcal/mol) and Kd values.

    ```bash
    cd PPAP/output
    ```

## Reference
Jie Qian, *et. al.* PPAP: A Protein-Protein Affinity Predictor Incorporating Interfacial Contact-aware Attention, *submitted*  
