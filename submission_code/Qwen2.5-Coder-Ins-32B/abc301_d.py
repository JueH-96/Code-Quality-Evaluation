# YOUR CODE HERE
import sys

def max_binary_value(S, N):
    max_value = -1
    length = len(S)
    
    def generate_binary(index, current_value):
        nonlocal max_value
        if index == length:
            if current_value <= N:
                max_value = max(max_value, current_value)
            return
        
        if S[index] == '0':
            generate_binary(index + 1, current_value * 2)
        elif S[index] == '1':
            generate_binary(index + 1, current_value * 2 + 1)
        else:
            generate_binary(index + 1, current_value * 2)
            generate_binary(index + 1, current_value * 2 + 1)
    
    generate_binary(0, 0)
    return max_value

def main():
    input = sys.stdin.read().split()
    S = input[0]
    N = int(input[1])
    result = max_binary_value(S, N)
    print(result)

if __name__ == "__main__":
    main()