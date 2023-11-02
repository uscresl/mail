# Notes on pre-trained model
Our pre-trained model, pre-collected dataset and the cached initial states for planning can be accessed throuth this google drive link: [Google Drive Link (14.3G)](https://drive.google.com/file/d/16KjI8ONMgfuUWMHWP1rT2x7B97n94BFo/view?usp=sharing)

Loading the pretraine edge and dynamics GNN will require a corresponding `best_state.json` that stores the model information. They can be downloaded at this google drive link: [Google Drive Link](https://drive.google.com/drive/folders/1CdCodTJ65NGK2qMK0wjsaOp_Ra6sTmD7?usp=sharing). Just put these json files under the same directory as the pre-trained model. 

Once uncompressed, you will find
* `1213_release_n1000.pkl`: The cached initial states for training.
* `dataset`: Pre-collected dataset for training.
* `vsbl_dyn_150.pth`: Pre-trained dynamics GNN for partially observed point cloud.
* `vsbl_edge_best.pth`: Pre-trained Edge GNN.
* `cloth_flatten_init_states_test_40.pkl` and `cloth_flatten_init_states_test_40_2.pkl`: in total 40 initial states for testing cloth smoothing on square clothes (each cached file has 20 states).
* `cloth_flatten_test_retangular_1.pkl` and `cloth_flatten_test_retangular_1.pkl`: Initial states for testing on rectangular clothes (each cached file has 20 states).
* `tshirt_flatten_init_states_small_2021_05_28_01_22.pkl`: Initial states for testing on t-shirt (20 states).

`tshirt_flatten_init_states_small_2021_05_28_01_16.pkl`: Initial states for testing on t-shirt using another 20 states can be downloaded via this [Google Drive Link](https://drive.google.com/drive/folders/1CdCodTJ65NGK2qMK0wjsaOp_Ra6sTmD7?usp=sharing)
