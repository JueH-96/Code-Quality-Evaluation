import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:]))
    
    def is_valid(seq):
        return sum(seq) % K == 0
    
    def backtrack(seq):
        if len(seq) == N:
            if is_valid(seq):
                print(" ".join(map(str, seq)))
            return
        
        for i in range(1, R[len(seq)] + 1):
            seq.append(i)
            backtrack(seq)
            seq.pop()
    
    backtrack([])

if __name__ == "__main__":
    solve()