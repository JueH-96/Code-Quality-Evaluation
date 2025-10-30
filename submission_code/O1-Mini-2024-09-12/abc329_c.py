# YOUR CODE HERE
import sys

def main():
    import sys
    N_and_S = sys.stdin.read().split()
    N = int(N_and_S[0])
    S = N_and_S[1]
    max_runs = [0]*26
    current_char = S[0]
    current_run = 1
    for c in S[1:]:
        if c == current_char:
            current_run +=1
        else:
            idx = ord(current_char) - ord('a')
            if current_run > max_runs[idx]:
                max_runs[idx] = current_run
            current_char = c
            current_run =1
    # Update the last run
    idx = ord(current_char) - ord('a')
    if current_run > max_runs[idx]:
        max_runs[idx] = current_run
    total = sum(max_runs)
    print(total)

if __name__ == "__main__":
    main()