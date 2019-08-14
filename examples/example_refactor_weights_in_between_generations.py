'''
val_loss: 0.0701
val_vae_reconstruction_loss: 0.0696
val_dnn_latent_layer_loss: 1.0122e-08
val_dnn_latent_mod_loss: 1.1921e-08
val_vae_latent_args_loss: 4.2640e-04

val_loss = val_vae_reconstruction_loss + val_dnn_latent_layer_loss + 		   val_dnn_latent_mod_loss + val_vae_latent_args_loss
'''
dnn_weights = []
dnn_kl_weights = []
vae_weights = []
vae_kl_weights = []
for chrom in generation:
	dnn_weight_ = chrom.val_loss / chrom.val_dnn_latent_mod_loss
	dnn_kl_weight_ = chrom.val_loss / chrom.val_dnn_latent_layer_loss
	vae_weight_ = chrom.val_loss / chrom.val_vae_reconstruction_loss
	vae_kl_weight_ = chrom.val_loss / chrom.val_vae_latent_args_loss

	dnn_weights.append(dnn_weight_)
	dnn_kl_weights.append(dnn_kl_weight_)
	vae_weights.append(vae_weight_)
	vae_kl_weights.append(vae_kl_weight_)

dnn_weight = np.median(dnn_weights)
dnn_kl_weight = np.median(dnn_kl_weights)
vae_weight = np.median(vae_weights)
vae_kl_weight = np.median(vae_kl_weights)

for chrom in new_generation:
	chrom.dnn_weight = dnn_weight
	chrom.dnn_kl_weight = dnn_kl_weight
	chrom.vae_weight = vae_weight
	chrom.vae_kl_weight = vae_kl_weight
