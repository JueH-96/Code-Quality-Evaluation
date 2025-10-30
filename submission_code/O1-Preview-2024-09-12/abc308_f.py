# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import heapq

    import sys

    sys.setrecursionlimit(1 << 25)

    N, M = map(int, sys.stdin.readline().split())

    P_j = list(map(int, sys.stdin.readline().split()))
    L_i = list(map(int, sys.stdin.readline().split()))
    D_i = list(map(int, sys.stdin.readline().split()))

    total_price = sum(P_j)

    events = []

    for i in range(M):
        events.append( (L_i[i], 0, D_i[i]) )  # Coupons: event_code 0

    for idx, p in enumerate(P_j):
        events.append( (p, 1, idx) )  # Items: event_code 1

    # Sort events in increasing order of time, 'coupon' events before 'item' events at the same time
    events.sort(key=lambda x: (x[0], x[1]))

    total_discount = 0
    available_coupons = []

    P_j_list = P_j  # For simplicity

    for event in events:
        t, event_code, value = event
        if event_code == 0:  # Coupon event
            D_i_value = value  # D_i
            L_i_value = t  # L_i
            heapq.heappush(available_coupons, (-D_i_value, L_i_value))
        else:  # Item event
            idx = value
            P_j_value = P_j_list[idx]
            while available_coupons:
                neg_D_i, L_i_value = heapq.heappop(available_coupons)
                D_i_value = -neg_D_i
                if L_i_value <= P_j_value:
                    discount = min(D_i_value, P_j_value -1)
                    total_discount += discount
                    break  # Coupon used
                # else discard coupon that cannot be used
    total_cost = total_price - total_discount
    print(total_cost)

if __name__ == "__main__":
    threading.Thread(target=main).start()