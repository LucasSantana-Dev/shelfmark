// shelfmark capture + handoff plugin for OpenCode.
//
// UNVERIFIED: written against the OpenCode plugin API (event, chat.message,
// tool.execute.before/after, experimental.chat.system.transform) but not yet
// run against a live OpenCode. Every call is best effort and never throws
// into OpenCode.
//
// Install: copy to .opencode/plugin/shelfmark.ts (project) or
// ~/.config/opencode/plugin/shelfmark.ts (global). Needs `shelfmark-hook` on PATH.
import { spawn } from "node:child_process"

const AGENT = "opencode"

function hook(event: string, payload: Record<string, unknown>): Promise<string> {
  return new Promise((resolve) => {
    try {
      const child = spawn("shelfmark-hook", ["--agent", AGENT, "--event", event], {
        stdio: ["pipe", "pipe", "ignore"],
      })
      let out = ""
      child.stdout.on("data", (d) => (out += d))
      child.on("error", () => resolve(""))
      child.on("close", () => resolve(out.trim()))
      child.stdin.end(JSON.stringify(payload))
    } catch {
      resolve("")
    }
  })
}

export const Shelfmark = async ({ directory }: { directory: string }) => {
  const started = new Set<string>()
  const children = new Set<string>() // subagent sessions never claim a handoff
  const pending = new Map<string, string>() // sessionID -> handoff text to inject once
  const toolArgs = new Map<string, unknown>() // callID -> args from tool.execute.before

  async function start(sessionID: string) {
    if (started.has(sessionID) || children.has(sessionID)) return
    started.add(sessionID)
    const text = await hook("session-start", { sessionID, directory })
    if (text) pending.set(sessionID, text)
  }

  return {
    event: async ({ event }: { event: any }) => {
      const info = event?.properties?.info
      if (event?.type === "session.created" && info?.id) {
        if (info.parentID) children.add(info.id)
        else await start(info.id)
      } else if (event?.type === "session.idle") {
        const sessionID = event.properties?.sessionID
        if (sessionID && !children.has(sessionID)) await hook("stop", { sessionID, directory })
      } else if (event?.type === "session.compacted") {
        const sessionID = event.properties?.sessionID
        if (sessionID && !children.has(sessionID)) await hook("pre-compact", { sessionID, directory })
      } else if (event?.type === "session.deleted" && info?.id) {
        if (!children.has(info.id)) await hook("session-end", { sessionID: info.id, directory })
        started.delete(info.id)
        children.delete(info.id)
        pending.delete(info.id)
      }
    },

    "chat.message": async (input: any, output: any) => {
      const sessionID = input?.sessionID
      if (!sessionID || children.has(sessionID)) return
      await start(sessionID)
      const parts = output?.parts ?? []
      const text = parts.filter((p: any) => p?.type === "text").map((p: any) => p.text).join("\n")
      if (text) await hook("user-prompt", { sessionID, directory, prompt: text })
    },

    "tool.execute.before": async (input: any, output: any) => {
      if (input?.callID) toolArgs.set(input.callID, output?.args)
    },

    "tool.execute.after": async (input: any, output: any) => {
      const sessionID = input?.sessionID
      const args = toolArgs.get(input?.callID)
      toolArgs.delete(input?.callID)
      if (!sessionID || children.has(sessionID)) return
      await hook("post-tool", {
        sessionID,
        directory,
        tool: input?.tool,
        args,
        output: { output: output?.output, title: output?.title, metadata: output?.metadata },
      })
    },

    "experimental.chat.system.transform": async (input: any, output: any) => {
      const sessionID = input?.sessionID
      const text = sessionID && pending.get(sessionID)
      if (text && Array.isArray(output?.system)) {
        output.system.push(text)
        pending.delete(sessionID)
      }
    },
  }
}
