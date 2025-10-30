import sys

class MinHeap:
    def __init__(self):
        self.heap = []
        self.pos = {}

    def insert(self, val, idx):
        self.heap.append((val, idx))
        pos_new = len(self.heap) - 1
        self.pos[idx] = pos_new
        self._sift_up(pos_new)

    def delete(self, idx):
        pos_idx = self.pos[idx]
        item = self.heap[pos_idx]
        n = len(self.heap)
        last_item = self.heap.pop()
        if n > 1:
            self.heap[pos_idx] = last_item
            self.pos[last_item[1]] = pos_idx
            self._heapify(pos_idx)
        del self.pos[item[1]]
        return item

    def delete_min(self):
        if not self.heap:
            raise ValueError("heap empty")
        min_item = self.heap[0]
        n = len(self.heap)
        last_item = self.heap.pop()
        if n > 1:
            self.heap[0] = last_item
            self.pos[last_item[1]] = 0
            self._sift_down(0)
        del self.pos[min_item[1]]
        return min_item

    def _sift_up(self, pos):
        while pos > 0:
            parent_pos = (pos - 1) // 2
            if self.heap[pos] < self.heap[parent_pos]:
                self.heap[pos], self.heap[parent_pos] = self.heap[parent_pos], self.heap[pos]
                self.pos[self.heap[pos][1]] = pos
                self.pos[self.heap[parent_pos][1]] = parent_pos
                pos = parent_pos
            else:
                break

    def _sift_down(self, pos):
        n = len(self.heap)
        while True:
            left = 2 * pos + 1
            right = 2 * pos + 2
            smallest = pos
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest != pos:
                self.heap[pos], self.heap[smallest] = self.heap[smallest], self.heap[pos]
                self.pos[self.heap[pos][1]] = pos
                self.pos[self.heap[smallest][1]] = smallest
                pos = smallest
            else:
                break

    def _heapify(self, pos):
        n = len(self.heap)
        if pos >= n:
            return
        parent_pos = (pos - 1) // 2 if pos > 0 else -1
        if parent_pos >= 0 and self.heap[pos] < self.heap[parent_pos]:
            self._sift_up(pos)
        else:
            self._sift_down(pos)

    def __len__(self):
        return len(self.heap)

    def get_min_value(self):
        if not self.heap:
            raise ValueError("heap empty")
        return self.heap[0][0]

data = sys.stdin.read().split()
index = 0
N = int(data[index])
K = int(data[index + 1])
Q = int(data[index + 2])
index += 3

min_heap = MinHeap()
max_heap = MinHeap()
location = [None] * (N + 1)
sum_top_k = 0

# Initial setup
for i in range(1, K + 1):
    min_heap.insert(0, i)
    location[i] = 0

for i in range(K + 1, N + 1):
    max_heap.insert(0, i)  # Store -value, but initial value is 0, so -0 is 0
    location[i] = 1

# Now process Q updates
for _ in range(Q):
    X = int(data[index])
    Y = int(data[index + 1])
    index += 2

    # Remove the old element and rebalance
    heap_type = location[X]
    if heap_type == 0:  # min_heap
        item = min_heap.delete(X)
        old_val = item[0]
        sum_top_k -= old_val
        if len(max_heap) > 0:
            max_item = max_heap.delete_min()  # max_item is (-val_max, idx_max)
            val_max = -max_item[0]
            idx_max = max_item[1]
            min_heap.insert(val_max, idx_max)
            sum_top_k += val_max
            location[idx_max] = 0
    elif heap_type == 1:  # max_heap
        max_item = max_heap.delete(X)  # max_item is (-old_val, X)
        if len(min_heap) > 0:
            min_item = min_heap.delete_min()
            min_val = min_item[0]
            min_idx = min_item[1]
            sum_top_k -= min_val
            max_heap.insert(-min_val, min_idx)
            location[min_idx] = 1

    # Insert the new element with value Y and index X
    if len(min_heap) < K:
        min_heap.insert(Y, X)
        sum_top_k += Y
        location[X] = 0
    else:
        if Y > min_heap.get_min_value():
            min_heap.insert(Y, X)
            sum_top_k += Y
            min_item = min_heap.delete_min()
            min_val = min_item[0]
            min_idx = min_item[1]
            sum_top_k -= min_val
            max_heap.insert(-min_val, min_idx)
            location[min_idx] = 1
        else:
            max_heap.insert(-Y, X)
            location[X] = 1
            if len(max_heap) > N - K:
                max_item = max_heap.delete_min()  # max_item is (-val_max, idx_max)
                val_max = -max_item[0]
                idx_max = max_item[1]
                min_heap.insert(val_max, idx_max)
                sum_top_k += val_max
                location[idx_max] = 0

    # Output f(A) after each update
    print(sum_top_k)