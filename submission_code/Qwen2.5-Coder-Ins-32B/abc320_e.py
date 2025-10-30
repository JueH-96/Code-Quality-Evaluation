# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    
    events = []
    index = 2
    for _ in range(M):
        T = int(input[index])
        W = int(input[index + 1])
        S = int(input[index + 2])
        events.append((T, W, S))
        index += 3
    
    people = [0] * N
    queue = []
    current_person = 0
    
    for T, W, S in events:
        while queue and queue[0][0] <= T:
            _, person = heapq.heappop(queue)
            current_person = person
        
        if current_person < N:
            people[current_person] += W
            heapq.heappush(queue, (T + S, current_person))
            current_person += 1
    
    for amount in people:
        print(amount)

if __name__ == "__main__":
    main()