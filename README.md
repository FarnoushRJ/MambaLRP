<center>
    <img src='assets/MambaLRP_logo.jpeg', width='1000'>
</center>

This is the official implementation of the paper "[MambaLRP: Explaining Selective State Space Sequence Models](https://arxiv.org/abs/2406.07592)".

```BibTeX
@misc{jafari2024mambalrp,
      title={MambaLRP: Explaining Selective State Space Sequence Models}, 
      author={Farnoush Rezaei Jafari and Grégoire Montavon and Klaus-Robert Müller and Oliver Eberle},
      year={2024},
      eprint={2406.07592},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
<p align="left">
  <img src="assets/about.png", height='50'/>
</p>

<p align="justify">Recent sequence modeling approaches using Selective State Space Sequence Models, referred to as Mamba models, have seen a surge of interest. These models allow efficient processing of long sequences in linear time and are rapidly being adopted in a wide range of applications such as language modeling, demonstrating promising performance. To foster their reliable use in real-world scenarios, it is crucial to augment their transparency. Our work bridges this critical gap by bringing explainability, particularly Layer-wise Relevance Propagation (LRP), to the Mamba architecture. Guided by the axiom of relevance conservation, we identify specific components in the Mamba architecture, which cause unfaithful explanations. To remedy this issue, we propose MambaLRP, a novel algorithm within the LRP framework, which ensures a more stable and reliable relevance propagation through these components. Our proposed method is theoretically sound and excels in achieving state-of-the-art explanation performance across a diverse range of models and datasets. Moreover, MambaLRP facilitates a deeper inspection of Mamba architectures, uncovering various biases and evaluating their significance. It also enables the analysis of previous speculations regarding the long-range capabilities of Mamba models.</p>

<p align="center">
  <img src="assets/results.svg", width='700'/>
</p>

<p align="left">
  <img src="assets/installation.png", height='50'/>
</p>

```python
!git clone https://github.com/FarnoushRJ/MambaLRP.git
!pip install -e ./MambaLRP --quiet
```

<p align="left">
  <img src="assets/todo.png", height='50'/>
</p>

- [ ] Add demo notebooks.
- [ ] Add LRP for Vision Mamba.

<p align="left">
  <img src="assets/acknowledgements.png", height='50'/>
</p>

- This repo is built using components from [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/mamba) and [Mamba](https://github.com/state-spaces/mamba).
- The logos are created using [Canva](https://www.canva.com/)
