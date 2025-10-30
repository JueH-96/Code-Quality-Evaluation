import sys

def main():
    input = sys.stdin.readline
    Q = int(input().strip())
    # Initialize stack with 100 cards labeled 0
    stack = [0] * 100
    outputs = []
    
    for _ in range(Q):
        parts = input().split()
        c = int(parts[0])
        if c == 1:
            # Push x onto the stack
            x = int(parts[1])
            stack.append(x)
        else:
            # Pop from the stack and record the value
            val = stack.pop()
            outputs.append(val)
    
    # Print all outputs for type-2 queries
    print("
".join(map(str, outputs)))

if __name__ == "__main__":
    main()