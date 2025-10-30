def get_inversion_number(p):
    n = len(p)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                inversions += 1
    return inversions

def solve():
    n = int(input())
    p = list(map(int, input().split()))
    m = int(input())
    a = list(map(int, input().split()))
    
    current_p = list(p)
    
    for k_val in a:
        for i in range(k_val - 1):
            if current_p[i] > current_p[i+1]:
                current_p[i], current_p[i+1] = current_p[i+1], current_p[i]
        inversion_count = get_inversion_number(current_p)
        print(inversion_count)

if __name__ == '__main__':
    solve()