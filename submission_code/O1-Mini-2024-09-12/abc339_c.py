# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    cumulative = 0
    min_cumulative = 0
    total_sum = 0
    
    for ai in A:
        cumulative += ai
        if cumulative < min_cumulative:
            min_cumulative = cumulative
    initial_S = max(0, -min_cumulative)
    total_sum = cumulative
    answer = initial_S + total_sum
    print(answer)

if __name__ == "__main__":
    main()