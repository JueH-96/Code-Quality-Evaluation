# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    
    # Initialize the result and the previous move of Takahashi
    result = 0
    prev_move = ''
    
    for i in range(N):
        if S[i] == 'R':
            if prev_move != 'P':
                result += 1
                prev_move = 'P'
        elif S[i] == 'P':
            if prev_move != 'S':
                result += 1
                prev_move = 'S'
        elif S[i] == 'S':
            if prev_move != 'R':
                result += 1
                prev_move = 'R'
    
    print(result)

if __name__ == "__main__":
    main()