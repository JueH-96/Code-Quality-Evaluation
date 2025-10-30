import heapq

class Node:
    def __init__(self, index):
        self.index = index
        self.prev = None
        self.next = None

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    events = []
    idx = 2
    for _ in range(M):
        T = int(data[idx])
        W = int(data[idx+1])
        S = int(data[idx+2])
        events.append((T, W, S))
        idx += 3
    
    # Initialize linked list
    head = Node(-1)  # Dummy head
    tail = Node(-1)  # Dummy tail
    head.next = tail
    tail.prev = head
    nodes = [None] * N
    current = head
    for i in range(N):
        node = Node(i)
        nodes[i] = node
        current.next = node
        node.prev = current
        current = node
    node.next = tail
    tail.prev = current
    
    # Min-heap for return times
    return_heap = []
    
    # Noodles received by each person
    noodles = [0] * N
    
    # Process each event
    for t, w, s in events:
        # Bring back people who have returned by time t
        while return_heap and return_heap[0][0] <= t:
            _, index = heapq.heappop(return_heap)
            node = nodes[index]
            # Insert back into the linked list at their original position
            prev_node = node.prev
            next_node = node.next
            if prev_node is not None and next_node is not None:
                prev_node.next = node
                next_node.prev = node
                node.prev = prev_node
                node.next = next_node
        
        # Serve the front person
        front = head.next
        if front != tail:
            person_index = front.index
            noodles[person_index] += w
            # Remove front person from the linked list
            front.prev.next = front.next
            front.next.prev = front.prev
            # Add to return heap
            return_time = t + s
            heapq.heappush(return_heap, (return_time, person_index))
    
    # Output the results
    for n in noodles:
        print(n)

if __name__ == '__main__':
    main()