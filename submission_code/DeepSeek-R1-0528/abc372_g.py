import heapq
import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); index += 1
        constraints = []
        x_max = 10**18
        for i in range(n):
            A = int(data[index]); B = int(data[index+1]); C = int(data[index+2]); index += 3
            num_bound = C - B - 1
            if num_bound < 0:
                bound = -1
            else:
                bound = num_bound // A
            if bound < x_max:
                x_max = bound
            constraints.append((A, B, C))
        
        if x_max < 1:
            results.append("0")
            continue
        
        if x_max * n <= 10000000:
            total = 0
            for x in range(1, x_max + 1):
                y_max = 10**18
                for (A, B, C_val) in constraints:
                    num = C_val - A * x - 1
                    if num < 0:
                        y_val = -10**18
                    else:
                        y_val = num // B
                    if y_val < y_max:
                        y_max = y_val
                if y_max > 0:
                    total += y_max
            results.append(str(total))
        else:
            total_events = 0
            for (A, B, C_val) in constraints:
                num_initial = C_val - A * 1 - 1
                if num_initial < 0:
                    initial_k_i = -10**18
                else:
                    initial_k_i = num_initial // B
                num_final = C_val - A * x_max - 1
                if num_final < 0:
                    final_k_i = -10**18
                else:
                    final_k_i = num_final // B
                if initial_k_i < 0:
                    initial_k_i = -10**18
                if final_k_i < 0:
                    final_k_i = -10**18
                drops = initial_k_i - final_k_i
                if drops < 0:
                    drops = 0
                total_events += drops

            if total_events > 10000000:
                initial_global_min = 10**18
                for (A, B, C_val) in constraints:
                    num = C_val - A * 1 - 1
                    if num < 0:
                        k_val_here = -10**18
                    else:
                        k_val_here = num // B
                    if k_val_here < initial_global_min:
                        initial_global_min = k_val_here
                if initial_global_min < 0:
                    results.append("0")
                else:
                    if initial_global_min > 10000000:
                        results.append("0")
                    else:
                        total = 0
                        k = 1
                        while k <= initial_global_min:
                            x_max_k = 10**18
                            for (A, B, C_val) in constraints:
                                num2 = C_val - 1 - k * B
                                if num2 < 0:
                                    x_val_here = -1
                                else:
                                    x_val_here = num2 // A
                                if x_val_here < x_max_k:
                                    x_max_k = x_val_here
                            if x_max_k < 1:
                                break
                            x_here = min(x_max, x_max_k)
                            total += x_here
                            k += 1
                        results.append(str(total))
            else:
                N_i = n
                INF = 10**18
                current_k = [0] * N_i
                next_event_val = [0] * N_i
                events = []
                heap_k = []
                for i in range(N_i):
                    A, B, C_val = constraints[i]
                    num = C_val - A * 1 - 1
                    if num < 0:
                        k_val_here = -10**18
                    else:
                        k_val_here = num // B
                    current_k[i] = k_val_here
                    num2 = C_val - 1 - current_k[i] * B
                    if num2 < 0:
                        next_x_here = x_max + 1
                    else:
                        next_x_here = num2 // A + 1
                        if next_x_here > x_max:
                            next_x_here = x_max + 1
                    next_event_val[i] = next_x_here
                    heapq.heappush(events, (next_x_here, i))
                    heapq.heappush(heap_k, (current_k[i], i))
                
                current_min = 0
                while heap_k:
                    val, idx = heap_k[0]
                    if val == current_k[idx]:
                        current_min = val
                        break
                    else:
                        heapq.heappop(heap_k)
                if not heap_k:
                    current_min = 0

                x_current = 1
                total_sum = 0
                while x_current <= x_max:
                    if events:
                        next_x_event = events[0][0]
                    else:
                        next_x_event = x_max + 1
                    if next_x_event > x_max:
                        next_x_event = x_max + 1
                    seg_length = next_x_event - x_current
                    total_sum += current_min * seg_length
                    x_current = next_x_event
                    if x_current > x_max:
                        break
                    to_update = []
                    while events and events[0][0] == x_current:
                        event_x, idx = heapq.heappop(events)
                        to_update.append(idx)
                    for idx in to_update:
                        current_k[idx] -= 1
                        A, B, C_val = constraints[idx]
                        num2 = C_val - 1 - current_k[idx] * B
                        if num2 < 0:
                            next_x_here = x_max + 1
                        else:
                            next_x_here = num2 // A + 1
                            if next_x_here > x_max:
                                next_x_here = x_max + 1
                        next_event_val[idx] = next_x_here
                        heapq.heappush(events, (next_x_here, idx))
                        heapq.heappush(heap_k, (current_k[idx], idx))
                    while heap_k:
                        val, idx = heap_k[0]
                        if val == current_k[idx]:
                            current_min = val
                            break
                        else:
                            heapq.heappop(heap_k)
                    if not heap_k:
                        current_min = 0
                results.append(str(total_sum))
    for res in results:
        print(res)

if __name__ == "__main__":
    main()