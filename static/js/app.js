/* =========================================================
   DesignScore UI behaviours
   - skeleton screen + top bar while a page loads
   - real upload progress bar for the analyse form
   - inline toasts for errors (no page reload needed)
   ========================================================= */
(function () {
    "use strict";

    /* ---------------- top loading bar ---------------- */
    function topBar() {
        var el = document.getElementById("ui-topbar");
        if (!el) {
            el = document.createElement("div");
            el.id = "ui-topbar";
            el.className = "ui-topbar";
            document.body.appendChild(el);
        }
        return el;
    }

    function navStart() {
        topBar().classList.add("active", "indeterminate");
        showSkeleton();
    }

    function navEnd() {
        topBar().classList.remove("active", "indeterminate");
        hideSkeleton();
    }

    /* ---------------- skeleton screen ---------------- */
    function showSkeleton() {
        if (document.getElementById("ui-skeleton")) return;
        var s = document.createElement("div");
        s.id = "ui-skeleton";
        s.className = "ui-skeleton";
        s.innerHTML =
            '<div class="ui-sk ui-sk-head"></div>' +
            '<div class="ui-sk-row">' +
              '<div class="ui-sk ui-sk-side"></div>' +
              '<div class="ui-sk-main">' +
                '<div class="ui-sk ui-sk-title"></div>' +
                '<div class="ui-sk-cards">' +
                  '<div class="ui-sk ui-sk-card"></div>' +
                  '<div class="ui-sk ui-sk-card"></div>' +
                  '<div class="ui-sk ui-sk-card"></div>' +
                '</div>' +
                '<div class="ui-sk ui-sk-block"></div>' +
              '</div>' +
            '</div>';
        document.body.appendChild(s);
    }

    function hideSkeleton() {
        var s = document.getElementById("ui-skeleton");
        if (s && s.parentNode) s.parentNode.removeChild(s);
    }

    /* Show structure immediately on internal navigation so a slow or
       failing network never shows a blank white screen. */
    document.addEventListener("click", function (ev) {
        var a = ev.target.closest ? ev.target.closest("a[href]") : null;
        if (!a) return;
        var href = a.getAttribute("href");
        if (!href || href.charAt(0) === "#") return;
        if (a.target === "_blank") return;
        if (a.hasAttribute("download") || a.dataset.noLoader) return;
        if (a.origin && a.origin !== window.location.origin) return;
        navStart();
    });

    window.addEventListener("pageshow", navEnd);
    window.addEventListener("pagehide", hideSkeleton);

    /* ---------------- toast ---------------- */
    function toast(message, kind) {
        var t = document.createElement("div");
        t.className = "ui-toast" + (kind === "ok" ? " ok" : "");
        t.textContent = message;
        document.body.appendChild(t);
        setTimeout(function () {
            t.classList.add("hide");
            setTimeout(function () {
                if (t.parentNode) t.parentNode.removeChild(t);
            }, 250);
        }, 4200);
    }
    window.uiToast = toast;

    /* ---------------- upload with progress ---------------- */
    var MAX_BYTES = 10 * 1024 * 1024;
    var OK_EXT = ["png", "jpg", "jpeg", "webp", "bmp", "gif"];

    function extOf(name) {
        var i = name.lastIndexOf(".");
        return i === -1 ? "" : name.slice(i + 1).toLowerCase();
    }

    var form = document.querySelector("form[data-upload]");
    if (form) {
        var input = form.querySelector('input[type="file"]');
        var bar = document.getElementById("ui-progress");
        var fill = document.getElementById("ui-progress-fill");
        var label = document.getElementById("ui-progress-label");
        var pct = document.getElementById("ui-progress-pct");
        var btn = form.querySelector("button[type=submit]");

        function setBar(value, text) {
            if (!bar) return;
            bar.classList.add("on");
            if (fill) fill.style.width = value + "%";
            if (label) label.textContent = text;
            if (pct) pct.textContent = value + "%";
        }

        form.addEventListener("submit", function (ev) {
            var file = input && input.files && input.files[0];
            if (!file) {
                ev.preventDefault();
                toast("Please choose a design to analyse.");
                return;
            }
            if (OK_EXT.indexOf(extOf(file.name)) === -1) {
                ev.preventDefault();
                toast("Unsupported file type. Use PNG, JPG, WEBP, BMP or GIF.");
                return;
            }
            if (file.size > MAX_BYTES) {
                ev.preventDefault();
                toast("That file is too large (max 10 MB).");
                return;
            }

            ev.preventDefault();
            if (btn) {
                btn.classList.add("ui-busy");
                btn.innerHTML = '<span class="ui-spin"></span> Analysing…';
            }
            setBar(0, "Uploading…");

            var data = new FormData(form);
            var xhr = new XMLHttpRequest();
            xhr.open("POST", form.action, true);
            xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");

            xhr.upload.onprogress = function (e) {
                if (e.lengthComputable) {
                    var p = Math.round((e.loaded / e.total) * 100);
                    setBar(p, p < 100 ? "Uploading…" : "Uploaded — analysing design…");
                }
            };

            xhr.onload = function () {
                setBar(100, "Analysing design…");
                var payload = null;
                try { payload = JSON.parse(xhr.responseText); } catch (err) { payload = null; }
                if (xhr.status >= 200 && xhr.status < 300 && payload && payload.ok) {
                    topBar().classList.add("active", "indeterminate");
                    window.location.href = payload.url;
                    return;
                }
                resetBtn();
                setBar(0, "");
                if (bar) bar.classList.remove("on");
                toast(payload && payload.message
                    ? payload.message
                    : "Something went wrong while analysing the design.");
            };

            xhr.onerror = function () {
                resetBtn();
                if (bar) bar.classList.remove("on");
                toast("Network problem: the design was not uploaded. Check your connection.");
            };

            xhr.ontimeout = function () {
                resetBtn();
                if (bar) bar.classList.remove("on");
                toast("The upload timed out. Please try again.");
            };

            xhr.send(data);
        });

        function resetBtn() {
            if (!btn) return;
            btn.classList.remove("ui-busy");
            btn.innerHTML = '<span>Analyse Design</span><span class="arrow">→</span>';
        }
    }
})();
