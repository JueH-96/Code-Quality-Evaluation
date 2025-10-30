# YOUR CODE HERE
def generate_sequences(n, k, r):
    def backtrack(index, current, total):
        if index == n:
            if total % k == 0:
                print(' '.join(map(str, current)))
            return
        for i in range(1, r[index] + 1):
            backtrack(index + 1, current + [i], total + i)
    
    backtrack(0, [], 0)

if __name__ == "__main__":
    n, k = map(int, input().split())
    r = list(map(int, input().split()))
    generate_sequences(n, k, r)