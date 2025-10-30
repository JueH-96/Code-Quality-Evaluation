import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A_list = list(map(int, sys.stdin.readline().split()))
    B_list = list(map(int, sys.stdin.readline().split()))

    A_list.sort()
    B_list.sort()

    ptrA = 0  # Pointer for sorted list of available boxes A
    ptrB = 0  # Pointer for sorted list of required minimums B
    total_cost = 0
    boxes_chosen_count = 0

    while ptrA < N and ptrB < M:
        if A_list[ptrA] >= B_list[ptrB]:
            # Box A_list[ptrA] can satisfy requirement B_list[ptrB].
            # Assign this box to this requirement.
            total_cost += A_list[ptrA]
            boxes_chosen_count += 1
            ptrA += 1  # Move to next box, as this one is "used"
            ptrB += 1  # Move to next requirement, as this one is "met"
        else: # A_list[ptrA] < B_list[ptrB]
            # Box A_list[ptrA] is too small for requirement B_list[ptrB].
            # Since B_list is sorted, B_list[ptrB] is the smallest current unmet requirement.
            # A_list[ptrA] cannot satisfy B_list[ptrB] or any subsequent, larger requirements B_list[ptrB+k].
            # Also, any boxes before A_list[ptrA] were either used for prior requirements 
            # or also deemed too small.
            # Therefore, A_list[ptrA] cannot be used for any of the M - ptrB people needing boxes.
            # We must skip A_list[ptrA] and try the next available box (A_list[ptrA+1]) for B_list[ptrB].
            ptrA += 1
            
    if boxes_chosen_count == M:
        print(total_cost)
    else:
        # Not enough boxes were found to satisfy all M requirements.
        print("-1")

solve()