import React from "react";
import ReactDOM from "react-dom/client";
import ChatWidget from "./ChatWidget";
import "./index.css";

// 既存のDOMに注入するためのルート要素を作成
const container = document.getElementById("chat-widget-root");
if (container) {
  ReactDOM.createRoot(container).render(
    <React.StrictMode>
      <ChatWidget />
    </React.StrictMode>
  );
}
