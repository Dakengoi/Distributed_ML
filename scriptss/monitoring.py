from prometheus_client import start_http_server, Gauge
import psutil
import time

cpu_usage_gauge = Gauge('cpu_usage', 'CPU usage percentage per node')
memory_usage_gauge = Gauge('memory_usage', 'Memory usage percentage per node')
network_traffic_gauge = Gauge('network_traffic', 'Total network traffic in bytes')
execution_time_gauge = Gauge('execution_time', 'Task execution time in seconds')
latency_gauge = Gauge('latency', 'Task processing delay in milliseconds')
job_completion_rate_gauge = Gauge('job_completion_rate', 'Percentage of successfully completed tasks')

start_time = time.time()


def collect_metrics():
    """Collects and updates Prometheus metrics."""
    while True:
        cpu_usage_gauge.set(psutil.cpu_percent())
        memory_usage_gauge.set(psutil.virtual_memory().percent)

        net_io = psutil.net_io_counters()
        total_bytes = net_io.bytes_sent + net_io.bytes_recv
        network_traffic_gauge.set(total_bytes)

        execution_time_gauge.set(time.time() - start_time)

        latency_gauge.set(psutil.cpu_percent() / 10)  # Example: Use CPU % as proxy

        job_completion_rate_gauge.set(95.0)

        time.sleep(2)


start_http_server(8000)
print("Prometheus metrics server started on http://localhost:8000")

collect_metrics()
