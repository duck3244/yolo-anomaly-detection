import axios from 'axios'

export const http = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

http.interceptors.response.use(
  (r) => r,
  (err) => {
    const msg = err?.response?.data?.detail || err?.message || 'Network error'
    return Promise.reject(new Error(msg))
  }
)
