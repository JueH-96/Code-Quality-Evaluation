import sys
from collections import deque

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it)); M = int(next(it)); L = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    C = [int(next(it)) for _ in range(L)]
    
    values = A + B + C
    total_cards = N + M + L
    
    if total_cards == 0:
        print("Aoki")
        return

    base_val = 1
    power = [0] * total_cards
    for i in range(total_cards):
        power[i] = base_val
        base_val *= 3

    mask0 = 0
    for i in range(total_cards):
        if i < N:
            st = 0
        elif i < N + M:
            st = 1
        else:
            st = 2
        mask0 += st * power[i]

    total_states = 2 * base_val

    def decode(mask_val, total_cards, power_arr):
        T_hand = set()
        A_hand = set()
        table_set = set()
        temp = mask_val
        for i in range(total_cards):
            r = temp % 3
            temp //= 3
            if r == 0:
                T_hand.add(i)
            elif r == 1:
                A_hand.add(i)
            else:
                table_set.add(i)
        return T_hand, A_hand, table_set

    moves = [[] for _ in range(total_states)]
    for state_index in range(total_states):
        base3 = base_val
        turn = state_index // base3
        mask_val = state_index % base3
        T_hand, A_hand, table_set = decode(mask_val, total_cards, power)
        if turn == 0:
            if not T_hand:
                continue
            for x in T_hand:
                next_mask = mask_val + 2 * power[x]
                next_state_index = 1 * base3 + next_mask
                moves[state_index].append(next_state_index)
                
                new_table = table_set | {x}
                for y in new_table:
                    if y == x:
                        continue
                    if values[y] < values[x]:
                        next_mask2 = mask_val + 2 * power[x] - 2 * power[y]
                        next_state_index2 = 1 * base3 + next_mask2
                        moves[state_index].append(next_state_index2)
        else:
            if not A_hand:
                continue
            for x in A_hand:
                next_mask = mask_val + 1 * power[x]
                next_state_index = 0 * base3 + next_mask
                moves[state_index].append(next_state_index)
                
                new_table = table_set | {x}
                for y in new_table:
                    if y == x:
                        continue
                    if values[y] < values[x]:
                        next_mask2 = mask_val + power[x] - power[y]
                        next_state_index2 = 0 * base3 + next_mask2
                        moves[state_index].append(next_state_index2)
    
    graph = [[] for _ in range(total_states)]
    for state_index in range(total_states):
        for next_index in moves[state_index]:
            graph[next_index].append(state_index)
            
    total_moves_arr = [len(moves[i]) for i in range(total_states)]
    dp = [None] * total_states
    count_arr = [0] * total_states
    Q = deque()
    
    for state_index in range(total_states):
        base3 = base_val
        turn = state_index // base3
        mask_val = state_index % base3
        T_hand, A_hand, table_set = decode(mask_val, total_cards, power)
        if (turn == 0 and not T_hand) or (turn == 1 and not A_hand):
            dp[state_index] = False
            Q.append(state_index)
            
    while Q:
        s = Q.popleft()
        for t in graph[s]:
            if dp[t] is not None:
                continue
            if not dp[s]:
                dp[t] = True
                Q.append(t)
            else:
                count_arr[t] += 1
                if count_arr[t] == total_moves_arr[t]:
                    dp[t] = False
                    Q.append(t)
                    
    initial_state_index = 0 * base_val + mask0
    if dp[initial_state_index]:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == '__main__':
    main()