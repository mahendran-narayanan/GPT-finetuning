import os
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from transformers.modeling_tf_utils import get_initializer

# Using 2 cores
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Using a pretrained model GPT2
# We can also use other variants of GPT

print("Downloading model")
model = TFGPT2LMHeadModel.from_pretrained("gpt2")

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load and preprocess text data
with open("data_input.txt", "r") as f:
    lines = f.read().split("\n")

# Encoding the data using the tokenizer
# Tokenizer truncate the sequences to a max_length of 1024 tokens
input_ids = []
for line in lines:
    encoding = tokenizer.encode(line, add_special_tokens=True, max_length=1024, truncation=True)
    input_ids.append(encoding)

# Parameters
batch_size = 2
num_epochs = 10
learning_rate = 5e-5

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Loss function
lossfunction = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Fine-tune the model using low-rank adaptation

for layer in model.transformer.h:
    layer.attention_output_dense = tf.keras.layers.Dense(units=256, kernel_initializer=get_initializer(0.02), name="attention_output_dense")

model.summary()

# Train the model
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size]
        # Padding the batch to maintain the same length
        batch = tf.keras.preprocessing.sequence.pad_sequences(batch, padding="post")
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        with tf.GradientTape() as tape:
            logits = model(inputs)[0]
            loss = lossfunction(targets, logits)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if i % (20 * batch_size) == 0:
            print(f"Batch {i}/{len(input_ids)} - loss: {loss:.4f}")

# Save the resultant output model
model.save_pretrained("tf_gpt2_keras_lora")

# Generate text

input_ids = tokenizer.encode("The generated sentence for the model is", return_tensors="tf")
output = model.generate(input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0], skip_special_tokens=True))