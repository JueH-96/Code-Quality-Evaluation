# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    cards = []
    for i in range(N):
        A = int(data[2*i + 1])
        C = int(data[2*i + 2])
        cards.append((A, C, i + 1))
    
    cards.sort(key=lambda x: (x[0], x[1]))
    
    stack = []
    for A, C, idx in cards:
        while stack and stack[-1][0] < A and stack[-1][1] > C:
            stack.pop()
        stack.append((A, C, idx))
    
    print(len(stack))
    print(' '.join(str(card[2]) for card in stack))

if __name__ == "__main__":
    main()