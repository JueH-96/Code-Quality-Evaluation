import sys
sys.setrecursionlimit(100000)

memo = {}
visiting = set()

def G(B):
    if not B:
        return 0
    key = tuple(sorted(B))
    if key in memo:
        return memo[key]
    if all(x == 1 for x in B):
        return len(B)
    if key in visiting:
        return 0
    
    visiting.add(key)
    evens = [x for x in B if x % 2 == 0]
    odds = [x for x in B if x % 2 == 1]
    list0 = sorted([(x + 1) // 2 for x in odds])
    list1 = sorted([(x + 2) // 2 for x in evens])
    val0 = G(list0)
    val1 = G(list1)
    res = len(B) - min(val0, val1)
    visiting.remove(key)
    memo[key] = res
    return res

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    global memo, visiting
    memo = {}
    visiting = set()
    result = G(A)
    print(result)

if __name__ == "__main__":
    main()