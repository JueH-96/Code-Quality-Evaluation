import bisect
import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    S = data[2+M:2+M+N]
    
    current_scores = []
    for i in range(N):
        solved = [A[j] for j in range(M) if S[i][j] == 'o']
        current_score = sum(solved) + (i + 1)
        current_scores.append(current_score)
    
    for i in range(N):
        other_scores = current_scores[:i] + current_scores[i+1:]
        other_max = max(other_scores)
        needed = other_max + 1
        current = current_scores[i]
        
        if current >= needed:
            print(0)
            continue
        
        unsolved = [A[j] for j in range(M) if S[i][j] == 'x']
        unsolved.sort(reverse=True)
        prefix_sum = list(itertools.accumulate(unsolved, initial=0))
        
        k = bisect.bisect_left(prefix_sum, needed - current)
        print(k)

if __name__ == "__main__":
    main()