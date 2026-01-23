import "./src/env.js";

/** @type {import("next").NextConfig} */
const config = {
  async rewrites() {
    return [
      {
        source: "/inference_url_here",
        destination:
          "https://jaswanth-s006--audio-cnn-inference-audioclassifier-inference.modal.run",
      },
    ];
  },
};

export default config;
