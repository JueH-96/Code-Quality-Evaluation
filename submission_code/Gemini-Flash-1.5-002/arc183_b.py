# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    groups = []
    visited = [False] * n
    for i in range(n):
        if visited[i]:
            continue
        group = []
        q = [i]
        visited[i] = True
        while q:
            curr = q.pop(0)
            group.append(curr)
            for j in range(max(0, curr - k), min(n, curr + k + 1)):
                if not visited[j]:
                    visited[j] = True
                    q.append(j)
        groups.append(group)

    for group in groups:
        group_a = [a[i] for i in group]
        group_b = [b[i] for i in group]
        
        group_a_counts = {}
        group_b_counts = {}
        for x in group_a:
            group_a_counts[x] = group_a_counts.get(x, 0) + 1
        for x in group_b:
            group_b_counts[x] = group_b_counts.get(x, 0) + 1

        
        possible = False
        for val_a in group_a_counts:
          possible_inner = True
          temp_group_b_counts = group_b_counts.copy()
          for i in group:
            if temp_group_b_counts.get(val_a,0) > 0:
              temp_group_b_counts[val_a] -=1
            else:
              possible_inner = False
              break
          if possible_inner and all(v == 0 for v in temp_group_b_counts.values()):
            possible = True
            break

        if not possible:
            print("No")
            return

    print("Yes")


t = int(input())
for _ in range(t):
    solve()