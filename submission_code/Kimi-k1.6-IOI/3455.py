class Solution:
    class Node:
        def __init__(self, val):
            self.val = val
            self.prev = None
            self.next = None
            self.prev_in_list = None
            self.next_in_list = None

    class CharList:
        def __init__(self):
            self.head = Solution.Node(None)
            self.tail = Solution.Node(None)
            self.head.next_in_list = self.tail
            self.tail.prev_in_list = self.head

        def append(self, node):
            last = self.tail.prev_in_list
            node.prev_in_list = last
            node.next_in_list = self.tail
            last.next_in_list = node
            self.tail.prev_in_list = node

        def remove(self, node):
            prev_node = node.prev_in_list
            next_node = node.next_in_list
            prev_node.next_in_list = next_node
            next_node.prev_in_list = prev_node
            node.prev_in_list = None
            node.next_in_list = None

    def minimumLength(self, s: str) -> int:
        # Build main linked list with sentinels
        head_main = self.Node(None)
        tail_main = self.Node(None)
        head_main.next = tail_main
        tail_main.prev = head_main

        current = head_main
        for char in s:
            new_node = self.Node(char)
            current.next = new_node
            new_node.prev = current
            current = new_node
        current.next = tail_main
        tail_main.prev = current

        # Build character lists
        from collections import defaultdict, deque
        char_lists = defaultdict(self.CharList)
        current = head_main.next
        while current != tail_main:
            char_lists[current.val].append(current)
            current = current.next

        # Initialize queue with eligible nodes
        queue = deque()
        in_queue = set()
        removed = set()

        current = head_main.next
        while current != tail_main:
            if current.val is None:
                current = current.next
                continue
            char_list = char_lists[current.val]
            # Check if current is not the first or last in its character list
            if current.prev_in_list != char_list.head and current.next_in_list != char_list.tail:
                queue.append(current)
                in_queue.add(current)
            current = current.next

        count = 0

        while queue:
            i = queue.popleft()
            if i in removed or i not in in_queue:  # i may have been removed or already processed
                if i in in_queue:
                    in_queue.remove(i)
                continue
            in_queue.remove(i)

            char_list = char_lists[i.val]
            # Re-check eligibility in case the list has changed since enqueued
            if i.prev_in_list == char_list.head or i.next_in_list == char_list.tail:
                continue

            # Proceed to remove j and k
            j = i.prev_in_list
            k = i.next_in_list

            # Remove j from main linked list
            j.prev.next = j.next
            j.next.prev = j.prev
            removed.add(j)
            char_list.remove(j)

            # Remove k from main linked list
            k.prev.next = k.next
            k.next.prev = k.prev
            removed.add(k)
            char_list.remove(k)

            count += 1

            # Check neighbors of j and k
            neighbors = [j.prev, j.next, k.prev, k.next]
            for x in neighbors:
                if x in (head_main, tail_main):
                    continue
                if x in removed:
                    continue
                if x.val is None:
                    continue
                x_char_list = char_lists[x.val]
                # Check if x is not first or last in its character list
                if x.prev_in_list != x_char_list.head and x.next_in_list != x_char_list.tail:
                    if x not in in_queue:
                        queue.append(x)
                        in_queue.add(x)

        return len(s) - 2 * count