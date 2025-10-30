import sys
import heapq

def flowing_noodles():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    events = []
    index = 2
    for _ in range(M):
        T = int(data[index])
        W = int(data[index + 1])
        S = int(data[index + 2])
        events.append((T, W, S))
        index += 3
    
    noodles_received = [0] * N
    current_line = list(range(N))
    out_people = []
    
    for T, W, S in events:
        # Return people who should be back in the line by time T
        while out_people and out_people[0][0] <= T:
            return_time, person_id = heapq.heappop(out_people)
            current_line.append(person_id)
        
        # If there's anyone in the line, the first person gets the noodles
        if current_line:
            person_id = current_line.pop(0)
            noodles_received[person_id] += W
            # They step out and will return at time T + S
            heapq.heappush(out_people, (T + S, person_id))
    
    for noodles in noodles_received:
        print(noodles)