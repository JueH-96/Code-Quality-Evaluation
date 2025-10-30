# YOUR CODE HERE
import sys

def convert_to_vertical(texts):
    max_length = max(len(text) for text in texts)
    vertical_texts = []

    for i in range(max_length):
        vertical_text = []
        for text in reversed(texts):
            if i < len(text):
                vertical_text.append(text[i])
            else:
                vertical_text.append('*')
        # Remove trailing '*'
        while vertical_text and vertical_text[-1] == '*':
            vertical_text.pop()
        vertical_texts.append(''.join(reversed(vertical_text)))

    return vertical_texts

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    texts = data[1:N+1]
    vertical_texts = convert_to_vertical(texts)
    for text in vertical_texts:
        print(text)

if __name__ == "__main__":
    main()