# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    S = input[0]
    T = input[1]
    
    if S == T:
        print(0)
        return
    
    steps = []
    S_list = list(S)
    
    for i in range(len(S)):
        if S[i] != T[i]:
            S_list[i] = T[i]
            steps.append(''.join(S_list))
    
    print(len(steps))
    for step in steps:
        print(step)

if __name__ == "__main__":
    main()