import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
  // Tauri's dev server must be on a fixed port.
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: { ignored: ["**/src-tauri/**"] },
  },

  build: {
    // Tauri's embedded WebView (WebView2/WebKit/WebKitGTK) supports ES2022+ natively.
    // Targeting esnext skips all downgrade transforms and eliminates polyfills.
    target: "esnext",

    // esbuild (default) is fast enough; no need for terser.
    minify: "esbuild",
    cssMinify: true,

    // Keep source maps out of the production bundle.
    sourcemap: false,

    // Warn when any individual chunk exceeds 200 kB (well under current size).
    chunkSizeWarningLimit: 200,

    rollupOptions: {
      output: {
        // Isolate the Tauri IPC runtime so it can be fingerprint-cached
        // independently of application code when the app grows.
        manualChunks: {
          "tauri-api": ["@tauri-apps/api"],
        },
      },
      // Tauri bundles everything into a self-contained WebView, so there are
      // no external scripts to mark as external here.
    },
  },

  // Inline assets smaller than 10 kB as data URIs (avoids extra HTTP requests
  // inside the Tauri asset protocol).
  assetsInclude: ["**/*.wgsl"],
  base: "./",
});
