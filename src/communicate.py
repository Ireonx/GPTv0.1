from functions import *
if __name__ == '__main__':
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print('loaded successfully!')
    m = model.to(device)
    while True:
        prompt = input("Prompt:\n")
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
        generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
        print(f'Completion:\n{generated_chars}')