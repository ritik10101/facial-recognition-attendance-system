// ============================================================
// app.js — Enhanced Modern UI + Webcam Logic (2025 Version)
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
  // ============================================================
  // THEME TOGGLE (light/dark)
  // ============================================================
  const themeToggle = document.querySelector(".theme-toggle");
  const savedTheme = localStorage.getItem("theme");

  if (savedTheme) {
    document.body.classList.toggle("dark", savedTheme === "dark");
  }

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const isDark = document.body.classList.toggle("dark");
      localStorage.setItem("theme", isDark ? "dark" : "light");

      // Rotate animation feedback
      themeToggle.style.transform = "rotate(180deg)";
      setTimeout(() => (themeToggle.style.transform = ""), 300);
    });
  }

  // ============================================================
  // USER MENU (avatar dropdown)
  // ============================================================
  const avatar = document.getElementById("user-avatar");
  const dropdown = document.getElementById("user-dropdown");

  if (avatar && dropdown) {
    avatar.addEventListener("click", (e) => {
      e.stopPropagation();
      dropdown.classList.toggle("show");
      dropdown.style.animation = "fadeIn 0.3s ease";
    });

    document.addEventListener("click", (e) => {
      if (!dropdown.contains(e.target) && e.target !== avatar) {
        dropdown.classList.remove("show");
      }
    });
  }

  // ============================================================
  // AUTO-HIDE TOAST MESSAGES (individual fade)
  // ============================================================
  const toasts = document.getElementById("toasts");
  if (toasts) {
    document.querySelectorAll(".toast").forEach((toast, i) => {
      setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(-10px)";
        setTimeout(() => toast.remove(), 500);
      }, 3000 + i * 200);
    });
  }

  // ============================================================
  // SMOOTH SCROLL TO TOP ON NAVIGATION
  // ============================================================
  document.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () =>
      window.scrollTo({ top: 0, behavior: "smooth" })
    );
  });

  // ============================================================
  // WEBCAM HANDLERS
  // ============================================================

  // ---------- USER LIVE ATTENDANCE ----------
  const videoMark = document.getElementById("video_mark");
  const overlayMark = document.getElementById("overlay_mark");
  const snapBtn = document.getElementById("snap");
  const statusEl = document.getElementById("status");

  if (videoMark && snapBtn && overlayMark) {
    const ctx = overlayMark.getContext("2d");

    // Start webcam
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        videoMark.srcObject = stream;
      })
      .catch((err) => {
        console.error("❌ Camera access denied:", err);
        if (statusEl) statusEl.textContent = "Camera not accessible.";
      });

    // Capture attendance snapshot
    snapBtn.addEventListener("click", async () => {
      overlayMark.width = videoMark.videoWidth;
      overlayMark.height = videoMark.videoHeight;
      ctx.drawImage(videoMark, 0, 0);
      const dataUrl = overlayMark.toDataURL("image/jpeg");
      if (statusEl) statusEl.textContent = "⏳ Processing...";

      try {
        const res = await fetch("/attendance/mark", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_base64: dataUrl.split(",")[1] }),
        });

        const data = await res.json();
        if (data.ok) {
          statusEl.textContent = "✅ " + data.message;
          statusEl.style.color = "green";
        } else {
          statusEl.textContent = "❌ " + (data.message || "Error");
          statusEl.style.color = "red";
        }
      } catch (err) {
        console.error(err);
        statusEl.textContent = "❌ Error sending request.";
        statusEl.style.color = "red";
      }
    });
  }

  // ---------- ADMIN LIVE CAPTURE ----------
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const startBtn = document.getElementById("startCapture");
  const progress = document.getElementById("progress");
  const preview = document.getElementById("preview");

  if (video && startBtn && overlay && progress) {
    const ctx = overlay.getContext("2d");
    let captureCount = 0;
    let totalCount = 0;
    let faceId = 1;

    // Initialize camera
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        video.srcObject = stream;
      })
      .catch((err) => {
        console.error("❌ Camera access denied:", err);
        alert("Camera not accessible. Please allow webcam permission.");
      });

    // Capture training images
    startBtn.addEventListener("click", async () => {
      const faceIdInput = document.getElementById("face_id");
      const countInput = document.getElementById("count");
      faceId = parseInt(faceIdInput?.value || "1");
      totalCount = parseInt(countInput?.value || "20");
      captureCount = 0;

      if (preview) preview.innerHTML = "";
      progress.value = 0;
      startBtn.disabled = true;
      const capturedImages = [];

      const captureInterval = setInterval(() => {
        if (captureCount >= totalCount) {
          clearInterval(captureInterval);
          progress.value = 100;
          startBtn.disabled = false;
          uploadCaptured(capturedImages);
          return;
        }

        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        const dataUrl = overlay.toDataURL("image/jpeg");
        capturedImages.push(dataUrl.split(",")[1]);
        captureCount++;
        progress.value = (captureCount / totalCount) * 100;

        const imgEl = document.createElement("img");
        imgEl.src = dataUrl;
        imgEl.width = 64;
        imgEl.height = 48;
        imgEl.style.margin = "3px";
        imgEl.style.borderRadius = "6px";
        imgEl.style.boxShadow = "0 1px 4px rgba(0,0,0,0.2)";
        preview?.appendChild(imgEl);
      }, 300);
    });

    // Upload captured images to backend
    async function uploadCaptured(images) {
      try {
        const res = await fetch("/admin/live_capture_upload", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ face_id: faceId, images }),
        });

        const data = await res.json();
        if (data.ok) {
          alert(
            `✅ ${data.saved} images saved.\nTraining: ${
              data.train_ok ? "✅ Done" : "❌ Failed"
            }`
          );
        } else {
          alert("❌ Upload failed: " + data.message);
        }
      } catch (err) {
        console.error("Upload error:", err);
        alert("❌ Upload error. Please retry.");
      }
    }
  }

  // ============================================================
  // OPTIONAL: Floating Admin Quick Actions (if admin)
  // ============================================================
  const isAdmin = document.querySelector("a[href='/admin']");
  if (isAdmin) {
    const panel = document.createElement("div");
    panel.innerHTML = `
      <div id="quick-admin-panel" style="
        position: fixed; bottom: 20px; right: 20px;
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 14px; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        padding: 10px; display: flex; flex-direction: column;
        gap: 6px; z-index: 1000; backdrop-filter: blur(10px);
      ">
        <button onclick="window.location.href='/admin/train'" class="btn small">⚙️ Train</button>
        <button onclick="window.location.href='/admin/upload'" class="btn small">📁 Upload</button>
        <button onclick="window.location.href='/admin/live_capture'" class="btn small">🎥 Capture</button>
      </div>
    `;
    document.body.appendChild(panel);
  }
});
