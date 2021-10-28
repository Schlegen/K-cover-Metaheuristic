def put_item(queue, item, size):
    if len(queue) < size:
        queue.append(item)
    else:
        queue.pop(0)
        queue.append(item)
    return queue

