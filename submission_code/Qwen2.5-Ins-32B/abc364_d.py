import sys

def find_kth_closest_distance(N, Q, a, b_k):
    results = []
    for b, k in b_k:
        distances = sorted([abs(a_i - b) for a_i in a])
        results.append(distances[k-1])
    return results

if __name__ == "__main__":
    input = sys.stdin.read
    N, Q, *data = map(int, input().split())
    a = data[:N]
    b_k = list(zip(data[N::2], data[N+1::2]))
    results = find_kth_closest_distance(N, Q, a, b_k)
    for result in results:
        print(result)