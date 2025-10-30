import heapq
import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        N = int(data[index]); index += 1
        constraints = []
        X_max = 10**18
        for i in range(N):
            A_i = int(data[index]); B_i = int(data[index+1]); C_i = int(data[index+2]); 
            index += 3
            constraints.append((A_i, B_i, C_i))
            x_i = (C_i - 1) // A_i
            if x_i < X_max:
                X_max = x_i
        
        if X_max == 0:
            results.append("0")
            continue
            
        y_val = [0] * N
        events = []
        current_min_y = 10**18
        
        for i in range(N):
            A_i, B_i, C_i = constraints[i]
            num = C_i - A_i * 1 - 1
            if num < 0:
                y_val[i] = -10**18
            else:
                y_val[i] = num // B_i
                next_x = (C_i - 1 - y_val[i] * B_i) // A_i + 1
                if next_x <= X_max:
                    heapq.heappush(events, (next_x, i))
                    
            if y_val[i] < current_min_y:
                current_min_y = y_val[i]
                
        if current_min_y < 0:
            results.append("0")
            continue
            
        ans = 0
        x = 1
        while x <= X_max:
            if current_min_y <= 0:
                break
                
            if events and events[0][0] == x:
                while events and events[0][0] == x:
                    next_x, i = heapq.heappop(events)
                    y_val[i] -= 1
                    if y_val[i] < current_min_y:
                        current_min_y = y_val[i]
                    if y_val[i] >= 0:
                        A_i, B_i, C_i = constraints[i]
                        next_x_new = (C_i - 1 - y_val[i] * B_i) // A_i + 1
                        if next_x_new <= X_max:
                            heapq.heappush(events, (next_x_new, i))
                            
            if events:
                next_event_x = events[0][0]
            else:
                next_event_x = X_max + 1
                
            seg_len = next_event_x - x
            ans += seg_len * current_min_y
            x = next_event_x
            
        results.append(str(ans))
        
    sys.stdout.write("
".join(results))

if __name__ == "__main__":
    main()