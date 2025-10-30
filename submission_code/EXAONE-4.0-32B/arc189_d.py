import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    left = [i-1 for i in range(n)]
    right = [i+1 for i in range(n)]
    total = a.copy()
    used = [False] * n
    stack = []
    for i in range(n):
        stack.append(i)
    
    while stack:
        i = stack.pop()
        if used[i]:
            continue
        while left[i] != -1 and not used[left[i]] and a[left[i]] < total[i]:
            total[i] += total[left[i]]
            used[left[i]] = True
            left[i] = left[left[i]]
            if left[i] != -1:
                right[left[i]] = i
        while right[i] != n and not used[right[i]] and a[right[i]] < total[i]:
            total[i] += total[right[i]]
            used[right[i]] = True
            right[i] = right[right[i]]
            if right[i] != n:
                left[right[i]] = i
        if not used[i]:
            stack.append(i)
    
    print(' '.join(map(str, total)))

if __name__ == "__main__":
    main()