class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        N = len(colors)
        colors += colors

        windows = []
        WindowManager = collections.namedtuple('WindowManager', ['record', 'count', 'start', 'end', 'change', 'update'])
        last_color = None
        start = 0
        change = defaultdict(int)
        for index, color in enumerate(colors):
            if color == last_color:
                if start >= index - 2:
                    start = index
                else:
                    windows.append(WindowManager(record=colors[start:index], count=index-start, start=start, end=index, change=change, update={}))

                change = defaultdict(int)
                change[color] = 1
                start = index
            
            else:
                change[color] += 1
            last_color = color

        currentPosition = -1

        def get_available_windows(window, size, mode=1):
            if window.count >= size:
                return [window.update | window.change]

            subWindows = []
            
            end = window.end + 1 
            currentPosition = window.end
            changes = window.change
            while True:
                # Reduce the start position until you have  size valid positions
                while end <= currentPosition + size - 1: 
                    if changes[colors[end]] > 0:
                        changes[colors[end]] -= 1
                    else:
                        changes[colors[end]] = 0
                    end += 1
                if end == N + N:
                    break
                while colors[currentPosition] not in changes or changes[colors[currentPosition]] == 0:
                    currentPosition -= 1
                changes[colors[currentPosition]] += 1

                if changes[colors[currentPosition + 1]] == 0:
                    changes[colors[currentPosition + 1]] = 1
                else:
                    changes[colors[currentPosition + 1]] += 1
                if changes[colors[currentPosition]] > 1:
                    del changes[colors[currentPosition]]
                else:
                    changes[colors[currentPosition]] = 1
                currentPosition += 1
                subWindows.append(window.update | changes)

            end = window.end
            start = window.start
            currentPosition = start
            while True:
                if end < start + size:
                    break
                while startPosition >= currentPosition + 1 and colors[currentPosition] not in changes and changes[colors[currentPosition+1]] > 0:
                    changes[colors[currentPosition+1]] -= 1
                    currentPosition += 1
                if changes[colors[currentPosition-1]] > 1:
                    del changes[colors[currentPosition-1]]
                if changes[colors[currentPosition-1]] == 0:
                    changes[colors[currentPosition-1]] = 0
                else:
                    changes[colors[currentPosition-1]] += 1
                if changes[colors[currentPosition]] == 0:
                    changes[colors[currentPosition]] = 1
                else:
                    changes[colors[currentPosition]] += 1
                subWindows.append(window.update | changes)

                currentPosition -= 1
                if currentPlayer == window.start:
                    break

            if mode == 1:
                return subWindows 
            else:
                next_start = currentPosition
                while next_start >= currentPosition + size:
                    pass


        def query_windows(size,mode=1):
            result = 0
            for window in windows:
                result += len(get_windows(window, size, mode=mode))
            return result
        
        def帮忙_change(index, newColor):
            index += N
            for window in windows:
                if window.start <= index < window.end:
                    windowManager.change[newColor] += 1
        

        result = []
        for q in queries:
            if q[0] == 1:
                result.append(query_windows(q[1]))
            else:
                帮忙_change(q[1], q[2])
        return result