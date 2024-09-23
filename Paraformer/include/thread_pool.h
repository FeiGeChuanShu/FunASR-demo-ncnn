#ifndef THREAD_POOL_H_
#define THREAD_POOL_H_
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <queue>
#include <atomic>

struct Task {
    int id;
    std::function<void()> func;
    Task(int id, std::function<void()> func):id(id),func(std::move(func)){}
};

class ThreadPool {
public:
    ThreadPool(size_t threads) : stop_(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->mtx_);
                        this->condition_.wait(
                            lock, [this] { return this->stop_ || !this->tasks_.empty(); });
                        if (this->stop_ && this->tasks_.empty())
                            return;
                        task = std::move(this->tasks_.front().func);
                        this->tasks_.pop();
                    }
                    task();
                }
                });
        }	
    }

    template <class F, class... Args>
    auto enqueue(F&& f, Args &&...args)
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mtx_);
            if (stop_)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks_.emplace(next_task_Id_++, [task]() {(*task)(); });
        }
        condition_.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_)
            worker.join();
    }

private:
    std::vector<std::thread> workers_;
    std::queue<Task> tasks_;
    std::mutex mtx_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    std::atomic<int> next_task_Id_{ 0 };
};


#endif
