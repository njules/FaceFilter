############
# Option 3 #
############

# young -> encoder_young -> generator_young2old -> rec_young2old -> encoder_old -> generator_old2young -> rec_old2young
#                                                           | -> discriminator_old (real/fake)                   |-> discriminatore_young (real/fake)

# create two dataloader one for old images and one for young images

# Step 1: 
# 
# requiresGrad(encoder_young, True)
# requiresGrad(generator_young2old, True)
# requiresGrad(discriminator_old, True)
# everything else does not require gradient
#
# young <- take young samples from the batch
# rec_young2old = generator_young2old(encoder_young(young))
# compute loss for discriminator_old (reduce loss on real old images and increase it on fake old)

# Step 2
#
# requiresGrad(encoder_old, True)
# requiresGrad(generator_old2young, True)
# requiresGrad(discriminator_young, True)
# everything else does not require gradient
#
# rec_old2young = generator_old2young(encoder_old(rec_young2old))
# compute loss for discriminator_young

# compute reconstruction (consistency) loss for the pair (young, rec_old2young)

# Step 3: repeat the previous two steps in the following direction: old --> young --> old
