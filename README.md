# Unimodal Face Classification with Multimodal Training

This is a PyTorch implementation of the following paper:

> Unimodal Face Classification with Multimodal Training
> Wenbin Teng (Boston University), Chongyang Bai (Dartmouth College)
> **Abstract:** *Face recognition is a crucial task in various multimedia applications such as security check, credential access and motion sensing games. However, the task is challenging when an input face is noisy (e.g. poor-condition RGB image) or lacks certain information (e.g. 3D face without color). In this work, we propose a Multimodal Training Unimodal Test (MTUT) framework for robust face classification, which exploits the cross-modality relationship during training and applies it as a complementary of the imperfect single modality input during testing. Technically, during training, the framework (1) builds both intra-modality and cross-modality autoencoders with the aid of facial attributes to learn latent embeddings as multimodal descriptors, (2) proposes a novel multimodal embedding divergence loss to align the heterogeneous features from different modalities, which also adaptively avoids the useless modality (if any) from confusing the model. This way, the learned autoencoders can generate robust embeddings in single-modality face classification on test stage. We evaluate our framework in two face classification datasets and two kinds of testing input: (1) poor-condition image and (2) point cloud or 3D face mesh, when both 2D and 3D modalities are available for training.*

The proposed method applies both 2D and 3D encoder to extract the embeddings of each individual modalities. Divergence between both embeddings is minimized adaptively through measuring the classification loss. Based on the type of testing modality, we use certain decoder to reconstruct 2D and 3D inputs from feature embeddings. An overview of the proposed network is shown in the following picture:
![Teaser image](./document/architecture.png)
