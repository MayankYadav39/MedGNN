import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react' // or vue, svelte, etc.

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    allowedHosts: [
      '7da6-103-159-214-186.ngrok-free.app'
      , 'e45a-103-159-214-186.ngrok-free.app'
      , '493d-103-159-214-189.ngrok-free.app',
      , '3dfd-103-159-214-189.ngrok-free.app '

    ]
  }
})
