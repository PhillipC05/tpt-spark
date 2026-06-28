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

    // Vite 8 uses oxc (via rolldown) — esbuild is no longer bundled.
    minify: "oxc",
    cssMinify: true,

    // Keep source maps out of the production bundle.
    sourcemap: false,

    // Warn when any individual chunk exceeds 200 kB (well under current size).
    chunkSizeWarningLimit: 200,

    rollupOptions: {
      output: {
        // Isolate the Tauri IPC runtime so it can be fingerprint-cached
        // independently of application code when the app grows.
        manualChunks: (id) => {
          if (id.includes("@tauri-apps/api")) return "tauri-api";
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
