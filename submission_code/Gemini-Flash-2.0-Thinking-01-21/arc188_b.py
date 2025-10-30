def solve():
    n, k = map(int, input().split())
    alice_sets = set()
    alice_sets.add(tuple(sorted([0])))
    if n % 2 == 0:
        alice_sets.add(tuple(sorted([n // 2])))
    for i in range(1, (n) // 2 + (1 if n % 2 != 0 else 0)):
        alice_sets.add(tuple(sorted([i, (n - i) % n])))
        
    bob_sets = set()
    if n % 2 != 0:
        bob_sets.add(tuple(sorted([k])))
    else:
        bob_sets.add(tuple(sorted([k])))
        bob_sets.add(tuple(sorted([(k + n // 2) % n])))
        
    for j in range(n):
        is_self_reflective_bob = False
        if n % 2 != 0:
            if j == k:
                is_self_reflective_bob = True
        else:
            if j == k or j == (k + n // 2) % n:
                is_self_reflective_bob = True
        if not is_self_reflective_bob:
            reflection_j = (2 * k - j) % n
            bob_sets.add(tuple(sorted([j, reflection_j])))
            
    distinct_sets_tuples = set()
    for s in alice_sets:
        distinct_sets_tuples.add(s)
    for s in bob_sets:
        distinct_sets_tuples.add(s)
        
    if len(distinct_sets_tuples) >= n:
        print("Yes")
    else:
        print("No")

t = int(input())
for _ in range(t):
    solve()